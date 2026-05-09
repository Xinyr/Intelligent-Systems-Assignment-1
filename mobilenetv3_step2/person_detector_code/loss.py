"""
loss.py  —  FCOS loss for single-class person detection.

Three terms:
    cls_loss  : Binary Focal Loss on every pixel
                (positives = pixels inside a GT person box)
    reg_loss  : GIoU Loss on positive pixels only
                (directly optimises box overlap, not L1 offsets)
    ctr_loss  : Binary Focal Loss for centerness on positive pixels

All three are normalised by the number of positive pixels across
the batch to keep magnitudes consistent.

Why GIoU instead of Smooth-L1?
    Smooth-L1 optimises each of l/t/r/b independently.
    GIoU directly maximises box overlap, which is exactly what
    we want — tighter crops for Model 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Focal loss (binary)
# ─────────────────────────────────────────────────────────────────

def focal_loss(pred_logit, target, alpha=0.25, gamma=2.0, reduction='sum'):
    """
    Binary focal loss.
    pred_logit : raw logit tensor  (any shape)
    target     : float 0/1 tensor (same shape)
    """
    p   = pred_logit.sigmoid()
    ce  = F.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    loss = a_t * (1 - p_t) ** gamma * ce
    if reduction == 'sum':  return loss.sum()
    if reduction == 'mean': return loss.mean()
    return loss


# ─────────────────────────────────────────────────────────────────
# GIoU loss
# ─────────────────────────────────────────────────────────────────

def giou_loss(pred_ltrb, gt_ltrb):
    """
    GIoU loss between predicted and GT (l,t,r,b) boxes.
    Both tensors shape (N, 4).  Values are distances in pixels.
    Returns mean loss over N.
    """
    # convert ltrb → x1y1x2y2  (relative, pixel units don't matter for IoU)
    def ltrb_to_xyxy(b):
        return torch.stack([-b[:,0], -b[:,1],  b[:,2],  b[:,3]], dim=1)

    p = ltrb_to_xyxy(pred_ltrb)
    g = ltrb_to_xyxy(gt_ltrb)

    # intersection
    ix1 = torch.max(p[:,0], g[:,0])
    iy1 = torch.max(p[:,1], g[:,1])
    ix2 = torch.min(p[:,2], g[:,2])
    iy2 = torch.min(p[:,3], g[:,3])
    iw  = (ix2 - ix1).clamp(min=0)
    ih  = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    # areas
    ap = (p[:,2]-p[:,0]).clamp(min=0) * (p[:,3]-p[:,1]).clamp(min=0)
    ag = (g[:,2]-g[:,0]).clamp(min=0) * (g[:,3]-g[:,1]).clamp(min=0)
    union = ap + ag - inter + 1e-6

    iou = inter / union

    # enclosing box
    ex1 = torch.min(p[:,0], g[:,0])
    ey1 = torch.min(p[:,1], g[:,1])
    ex2 = torch.max(p[:,2], g[:,2])
    ey2 = torch.max(p[:,3], g[:,3])
    enclose = ((ex2-ex1).clamp(min=0) * (ey2-ey1).clamp(min=0)) + 1e-6

    giou = iou - (enclose - union) / enclose
    return (1 - giou).mean()


# ─────────────────────────────────────────────────────────────────
# Target assignment
# ─────────────────────────────────────────────────────────────────

def assign_targets(preds, gt_boxes_list, input_h, input_w):
    """
    Assign ground-truth boxes to FCOS feature map pixels.

    Rules (standard FCOS):
      1. A pixel is positive if it falls inside at least one GT box.
      2. If a pixel falls inside multiple GT boxes, assign the
         smallest one (resolves ambiguity cleanly).
      3. Each FPN level handles boxes in a size range
         (regress_range below).

    Args:
        preds         : list of level-dicts from model.forward()
        gt_boxes_list : list (len=B) of (N,4) tensors  cx cy w h  [0,1]
        input_h/w     : model input resolution in pixels

    Returns:
        targets : list of level-dicts, each with:
            'cls_target'  (B, 1, H, W)  float 0/1
            'reg_target'  (B, 4, H, W)  l t r b in pixels
            'ctr_target'  (B, 1, H, W)  centerness [0,1]
            'pos_mask'    (B, H, W)     bool
    """
    device = preds[0]['cls'].device

    # size ranges per FPN level (pixels, in original image space)
    regress_ranges = [
        (0,   64),    # P3  stride 16  — small persons
        (64,  128),   # P4  stride 32
        (128, 256),   # P5  stride 32
        (256, 1e8),   # P6  stride 64  — large persons
    ]

    targets = []

    for level, (pred, (r_min, r_max)) in enumerate(
            zip(preds, regress_ranges)):

        H, W   = pred['cls'].shape[-2:]
        stride = pred['stride']
        B      = pred['cls'].size(0)

        # pixel centres in original image coordinates
        ys = (torch.arange(H, device=device).float() + 0.5) * stride
        xs = (torch.arange(W, device=device).float() + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)

        cls_t = torch.zeros(B, 1, H, W, device=device)
        reg_t = torch.zeros(B, 4, H, W, device=device)
        ctr_t = torch.zeros(B, 1, H, W, device=device)
        pos_m = torch.zeros(B, H, W, dtype=torch.bool, device=device)

        for b, gt_boxes in enumerate(gt_boxes_list):
            if gt_boxes.numel() == 0:
                continue

            gt_boxes = gt_boxes.to(device)

            # convert cx cy w h → x1 y1 x2 y2  in pixel space
            gx1 = (gt_boxes[:,0] - gt_boxes[:,2]/2) * input_w
            gy1 = (gt_boxes[:,1] - gt_boxes[:,3]/2) * input_h
            gx2 = (gt_boxes[:,0] + gt_boxes[:,2]/2) * input_w
            gy2 = (gt_boxes[:,1] + gt_boxes[:,3]/2) * input_h
            # (N,4)
            boxes_px = torch.stack([gx1,gy1,gx2,gy2],dim=1)

            # for each pixel, compute ltrb to every GT box
            # grid_x/y: (H,W)  →  (H,W,1)
            cx = grid_x.unsqueeze(-1)    # (H,W,1)
            cy = grid_y.unsqueeze(-1)

            l = cx - gx1[None,None,:]   # (H,W,N)
            t = cy - gy1[None,None,:]
            r = gx2[None,None,:] - cx
            b_ = gy2[None,None,:] - cy

            # pixel must be INSIDE the GT box
            inside = (l > 0) & (t > 0) & (r > 0) & (b_ > 0)  # (H,W,N)

            # max regression target must be within level's size range
            max_reg = torch.stack([l,t,r,b_],dim=-1).max(dim=-1).values  # (H,W,N)
            in_range = (max_reg >= r_min) & (max_reg <= r_max)

            valid = inside & in_range   # (H,W,N)

            if not valid.any():
                continue

            # area of each GT box (smaller box gets priority)
            areas = (boxes_px[:,2]-boxes_px[:,0]) * (boxes_px[:,3]-boxes_px[:,1])
            areas = areas[None,None,:].expand(H,W,-1)                 # (H,W,N)
            areas_masked = torch.where(valid, areas,
                                       torch.full_like(areas, 1e8))

            # assign each pixel to its smallest valid GT box
            min_area, best_gt = areas_masked.min(dim=-1)              # (H,W)
            pixel_positive = valid.any(dim=-1) & (min_area < 1e8)     # (H,W)

            if not pixel_positive.any():
                continue

            # gather ltrb for the assigned GT box
            idx = best_gt.unsqueeze(-1)                        # (H,W,1)
            l_a = torch.gather(l,  2, idx).squeeze(-1)        # (H,W)
            t_a = torch.gather(t,  2, idx).squeeze(-1)
            r_a = torch.gather(r,  2, idx).squeeze(-1)
            b_a = torch.gather(b_, 2, idx).squeeze(-1)

            # centerness = sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )
            ctr = torch.sqrt(
                (torch.min(l_a,r_a) / (torch.max(l_a,r_a) + 1e-6)) *
                (torch.min(t_a,b_a) / (torch.max(t_a,b_a) + 1e-6))
            ).clamp(0,1)

            # write targets
            cls_t[b, 0][pixel_positive] = 1.0
            reg_t[b, 0][pixel_positive] = l_a[pixel_positive]
            reg_t[b, 1][pixel_positive] = t_a[pixel_positive]
            reg_t[b, 2][pixel_positive] = r_a[pixel_positive]
            reg_t[b, 3][pixel_positive] = b_a[pixel_positive]
            ctr_t[b, 0][pixel_positive] = ctr[pixel_positive]
            pos_m[b][pixel_positive]    = True

        targets.append({
            'cls_target': cls_t,
            'reg_target': reg_t,
            'ctr_target': ctr_t,
            'pos_mask':   pos_m,
        })

    return targets


# ─────────────────────────────────────────────────────────────────
# FCOS loss
# ─────────────────────────────────────────────────────────────────

class FCOSLoss(nn.Module):
    """
    Combined FCOS loss.

    Args:
        cls_weight  : weight for classification term
        reg_weight  : weight for regression (GIoU) term
        ctr_weight  : weight for centerness term
        focal_alpha : focal loss alpha
        focal_gamma : focal loss gamma
    """
    def __init__(self,
                 cls_weight=1.0,
                 reg_weight=1.0,
                 ctr_weight=1.0,
                 focal_alpha=0.25,
                 focal_gamma=2.0):
        super().__init__()
        self.cls_w = cls_weight
        self.reg_w = reg_weight
        self.ctr_w = ctr_weight
        self.alpha  = focal_alpha
        self.gamma  = focal_gamma

    def forward(self, preds, gt_boxes_list, input_h, input_w):
        """
        Args:
            preds         : list of level-dicts from model.forward()
            gt_boxes_list : list (len=B) of (N,4) cx cy w h [0,1] tensors
            input_h/w     : model input resolution

        Returns:
            total_loss, cls_loss, reg_loss, ctr_loss  — scalar tensors
        """
        targets = assign_targets(preds, gt_boxes_list, input_h, input_w)

        total_cls = total_reg = total_ctr = 0.0
        total_pos = 0

        for pred, tgt in zip(preds, targets):
            pos_mask = tgt['pos_mask']               # (B,H,W)
            num_pos  = pos_mask.sum().item()
            total_pos += num_pos

            # ── classification loss (all pixels) ─────────────────
            cls_loss = focal_loss(
                pred['cls'].squeeze(1),              # (B,H,W)
                tgt['cls_target'].squeeze(1),
                self.alpha, self.gamma,
            )
            total_cls += cls_loss

            if num_pos > 0:
                # ── regression loss (positive pixels only) ────────
                pred_reg = pred['reg'].permute(0,2,3,1)[pos_mask]  # (P,4)
                gt_reg   = tgt['reg_target'].permute(0,2,3,1)[pos_mask]
                total_reg += giou_loss(pred_reg, gt_reg) * num_pos

                # ── centerness loss (positive pixels only) ────────
                ctr_loss = focal_loss(
                    pred['ctr'].squeeze(1)[pos_mask],
                    tgt['ctr_target'].squeeze(1)[pos_mask],
                    alpha=0.5, gamma=0.0,   # balanced for centerness
                )
                total_ctr += ctr_loss

        # normalise by total positives across all levels & batch
        denom = max(total_pos, 1)

        cls_l = self.cls_w * total_cls / denom
        reg_l = self.reg_w * (total_reg / denom if total_pos > 0
                               else torch.tensor(0.0, device=preds[0]['cls'].device))
        ctr_l = self.ctr_w * total_ctr / denom

        total = cls_l + reg_l + ctr_l
        return total, cls_l.detach(), reg_l.detach(), ctr_l.detach()


# ─────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from model import PersonDetector
    model = PersonDetector(pretrained=False)
    x     = torch.zeros(2, 3, 640, 640)
    preds = model(x)

    gt = [
        torch.tensor([[0.5, 0.5, 0.4, 0.8]]),
        torch.tensor([[0.3, 0.4, 0.2, 0.5], [0.7, 0.6, 0.15, 0.3]]),
    ]
    criterion = FCOSLoss()
    total, cls, reg, ctr = criterion(preds, gt, 640, 640)
    print(f'total={total:.4f}  cls={cls:.4f}  reg={reg:.4f}  ctr={ctr:.4f}')
    print('loss.py  OK')
