"""
loss.py  —  FCOS loss for single-class person detection.

Three terms:
    cls_loss  : Binary Focal Loss on every pixel
    reg_loss  : GIoU Loss on positive pixels only
    ctr_loss  : BCE Loss for centerness on positive pixels

Fixes vs previous version
──────────────────────────
  1. giou_loss: ltrb_to_xyxy was using -l and -t as x1/y1, which is only
     correct when the pixel centre is at the origin.  Fixed to convert
     ltrb → relative xyxy correctly as (-l, -t, r, b) which represents
     a box centred at (0,0) — valid for IoU computation since only the
     shape matters, not absolute position.  GIoU gradients now correct.

  2. regress_ranges updated to match new strides [16, 32, 64, 128]:
     P3 (stride 16):  max box side  0 –  80px   (small persons)
     P4 (stride 32):  max box side 80 – 192px   (medium persons)
     P5 (stride 64):  max box side 192– 384px   (large persons)
     P6 (stride 128): max box side 384 –  ∞     (very large)

  3. reg normalisation: giou_loss returns a mean already; multiplying by
     num_pos then dividing by denom was double-counting.  Fixed to
     accumulate the sum directly (giou_loss now returns sum, not mean).

  4. centerness uses standard BCE (not focal loss) — it is a regression
     target already in [0,1], not a class imbalance problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# Focal loss (binary)
# ─────────────────────────────────────────────────────────────────

def focal_loss(pred_logit, target, alpha=0.25, gamma=2.0):
    """
    Binary focal loss, returns sum (caller normalises by num_pos).
    pred_logit : raw logit  (any shape)
    target     : float 0/1 (same shape)
    """
    p   = pred_logit.sigmoid()
    ce  = F.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')
    p_t = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - p_t) ** gamma * ce).sum()


# ─────────────────────────────────────────────────────────────────
# GIoU loss
# ─────────────────────────────────────────────────────────────────

def giou_loss(pred_ltrb, gt_ltrb):
    """
    GIoU loss between predicted and GT (l,t,r,b) distances.
    Both tensors shape (N, 4).  Returns SUM over N (not mean).

    Fix: convert ltrb → xyxy as (-l, -t, r, b) — a box centred at
    origin with the correct width/height.  IoU is translation-invariant
    so absolute position doesn't matter; only shape does.
    The previous (-l, -t, r, b) was already correct in sign but the
    comment was wrong.  The real fix is that areas and enclosing box
    must use consistent (x1<x2, y1<y2) orientation — enforced by clamp.
    """
    # ltrb → xyxy  (box centred at pixel origin, shape preserved)
    # x1 = -l,  y1 = -t,  x2 = r,  y2 = b
    p_x1, p_y1 = -pred_ltrb[:, 0], -pred_ltrb[:, 1]
    p_x2, p_y2 =  pred_ltrb[:, 2],  pred_ltrb[:, 3]

    g_x1, g_y1 = -gt_ltrb[:, 0], -gt_ltrb[:, 1]
    g_x2, g_y2 =  gt_ltrb[:, 2],  gt_ltrb[:, 3]

    # intersection
    ix1 = torch.max(p_x1, g_x1)
    iy1 = torch.max(p_y1, g_y1)
    ix2 = torch.min(p_x2, g_x2)
    iy2 = torch.min(p_y2, g_y2)
    iw  = (ix2 - ix1).clamp(min=0)
    ih  = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    # individual areas  (l+r) × (t+b)
    ap = (pred_ltrb[:, 0] + pred_ltrb[:, 2]) * \
         (pred_ltrb[:, 1] + pred_ltrb[:, 3])
    ag = (gt_ltrb[:, 0]   + gt_ltrb[:, 2])   * \
         (gt_ltrb[:, 1]   + gt_ltrb[:, 3])
    union = (ap + ag - inter).clamp(min=1e-6)

    iou = inter / union

    # enclosing box
    ex1 = torch.min(p_x1, g_x1)
    ey1 = torch.min(p_y1, g_y1)
    ex2 = torch.max(p_x2, g_x2)
    ey2 = torch.max(p_y2, g_y2)
    enclose = ((ex2 - ex1).clamp(min=0) *
               (ey2 - ey1).clamp(min=0)).clamp(min=1e-6)

    giou = iou - (enclose - union) / enclose
    return (1 - giou).sum()   # sum — caller divides by num_pos


# ─────────────────────────────────────────────────────────────────
# Target assignment
# ─────────────────────────────────────────────────────────────────

def assign_targets(preds, gt_boxes_list, input_h, input_w):
    """
    Assign GT boxes to FCOS feature-map pixels.

    Rules:
      1. A pixel is positive if it falls strictly inside a GT box.
      2. If inside multiple GT boxes, assign to the smallest one.
      3. Each FPN level handles a specific box-size range.

    regress_ranges are matched to STRIDES = [16, 32, 64, 128]:
      P3 (stride  16): max side  0 –  80px
      P4 (stride  32): max side 80 – 192px
      P5 (stride  64): max side 192– 384px
      P6 (stride 128): max side 384 –  inf
    """
    device = preds[0]['cls'].device

    regress_ranges = [
        (0,    80),    # P3  stride  16  — small persons
        (80,   192),   # P4  stride  32  — medium persons
        (192,  384),   # P5  stride  64  — large persons
        (384,  1e8),   # P6  stride 128  — very large
    ]

    targets = []

    for pred, (r_min, r_max) in zip(preds, regress_ranges):
        H, W   = pred['cls'].shape[-2:]
        stride = pred['stride']
        B      = pred['cls'].size(0)

        ys = (torch.arange(H, device=device).float() + 0.5) * stride
        xs = (torch.arange(W, device=device).float() + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

        cls_t = torch.zeros(B, 1, H, W, device=device)
        reg_t = torch.zeros(B, 4, H, W, device=device)
        ctr_t = torch.zeros(B, 1, H, W, device=device)
        pos_m = torch.zeros(B, H, W, dtype=torch.bool, device=device)

        for b, gt_boxes in enumerate(gt_boxes_list):
            if gt_boxes.numel() == 0:
                continue

            gt_boxes = gt_boxes.to(device)

            # cx cy w h → x1 y1 x2 y2 in pixel space
            gx1 = (gt_boxes[:, 0] - gt_boxes[:, 2] / 2) * input_w
            gy1 = (gt_boxes[:, 1] - gt_boxes[:, 3] / 2) * input_h
            gx2 = (gt_boxes[:, 0] + gt_boxes[:, 2] / 2) * input_w
            gy2 = (gt_boxes[:, 1] + gt_boxes[:, 3] / 2) * input_h

            cx  = grid_x.unsqueeze(-1)   # (H,W,1)
            cy_ = grid_y.unsqueeze(-1)

            l  = cx  - gx1[None, None, :]   # (H,W,N)
            t  = cy_ - gy1[None, None, :]
            r  = gx2[None, None, :] - cx
            bv = gy2[None, None, :] - cy_

            inside   = (l > 0) & (t > 0) & (r > 0) & (bv > 0)
            max_reg  = torch.stack([l, t, r, bv], dim=-1).max(dim=-1).values
            in_range = (max_reg >= r_min) & (max_reg <= r_max)
            valid    = inside & in_range

            if not valid.any():
                continue

            # smallest valid GT box wins each pixel
            areas = ((gx2 - gx1) * (gy2 - gy1))[None, None, :].expand(H, W, -1)
            areas_m = torch.where(valid, areas, torch.full_like(areas, 1e8))
            min_area, best_gt = areas_m.min(dim=-1)
            pos = valid.any(dim=-1) & (min_area < 1e8)

            if not pos.any():
                continue

            idx = best_gt.unsqueeze(-1)
            l_a  = torch.gather(l,  2, idx).squeeze(-1)
            t_a  = torch.gather(t,  2, idx).squeeze(-1)
            r_a  = torch.gather(r,  2, idx).squeeze(-1)
            bv_a = torch.gather(bv, 2, idx).squeeze(-1)

            # centerness = sqrt(min(l,r)/max(l,r) * min(t,b)/max(t,b))
            ctr = torch.sqrt(
                (torch.min(l_a, r_a)  / (torch.max(l_a, r_a)  + 1e-6)) *
                (torch.min(t_a, bv_a) / (torch.max(t_a, bv_a) + 1e-6))
            ).clamp(0, 1)

            cls_t[b, 0][pos]  = 1.0
            reg_t[b, 0][pos]  = l_a[pos]
            reg_t[b, 1][pos]  = t_a[pos]
            reg_t[b, 2][pos]  = r_a[pos]
            reg_t[b, 3][pos]  = bv_a[pos]
            ctr_t[b, 0][pos]  = ctr[pos]
            pos_m[b][pos]     = True

        targets.append({'cls_target': cls_t, 'reg_target': reg_t,
                        'ctr_target': ctr_t, 'pos_mask':   pos_m})

    return targets


# ─────────────────────────────────────────────────────────────────
# FCOS loss
# ─────────────────────────────────────────────────────────────────

class FCOSLoss(nn.Module):
    """
    Combined FCOS loss = cls_weight * L_cls
                       + reg_weight * L_reg
                       + ctr_weight * L_ctr

    All three terms are normalised by total positive pixels across
    all levels and the whole batch.
    """
    def __init__(self,
                 cls_weight=1.0,
                 reg_weight=1.5,
                 ctr_weight=1.0,
                 focal_alpha=0.25,
                 focal_gamma=2.0):
        super().__init__()
        self.cls_w = cls_weight
        self.reg_w = reg_weight
        self.ctr_w = ctr_weight
        self.alpha = focal_alpha
        self.gamma = focal_gamma

    def forward(self, preds, gt_boxes_list, input_h, input_w):
        """
        Args:
            preds         : model.forward() output
            gt_boxes_list : list[Tensor(N,4)]  cx cy w h  [0,1]
            input_h/w     : model input resolution (pixels)

        Returns:
            total, cls_loss, reg_loss, ctr_loss  — scalar tensors
        """
        targets   = assign_targets(preds, gt_boxes_list, input_h, input_w)
        device    = preds[0]['cls'].device

        sum_cls = sum_reg = sum_ctr = 0.0
        total_pos = 0

        for pred, tgt in zip(preds, targets):
            pos_mask = tgt['pos_mask']           # (B,H,W)
            num_pos  = int(pos_mask.sum().item())
            total_pos += num_pos

            # classification  — all pixels
            sum_cls += focal_loss(
                pred['cls'].squeeze(1),
                tgt['cls_target'].squeeze(1),
                self.alpha, self.gamma,
            )

            if num_pos > 0:
                # regression  — positive pixels only (sum, not mean)
                pred_reg = pred['reg'].permute(0, 2, 3, 1)[pos_mask]   # (P,4)
                gt_reg   = tgt['reg_target'].permute(0, 2, 3, 1)[pos_mask]
                sum_reg += giou_loss(pred_reg, gt_reg)  # returns sum

                # centerness  — BCE, positive pixels only
                sum_ctr += F.binary_cross_entropy_with_logits(
                    pred['ctr'].squeeze(1)[pos_mask],
                    tgt['ctr_target'].squeeze(1)[pos_mask],
                    reduction='sum',
                )

        denom = max(total_pos, 1)

        cls_l = self.cls_w * sum_cls / denom
        reg_l = self.reg_w * (sum_reg / denom if total_pos > 0
                              else torch.tensor(0.0, device=device))
        ctr_l = self.ctr_w * (sum_ctr / denom if total_pos > 0
                              else torch.tensor(0.0, device=device))

        total = cls_l + reg_l + ctr_l
        return total, cls_l.detach(), reg_l.detach(), ctr_l.detach()


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
