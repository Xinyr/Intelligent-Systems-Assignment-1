"""
model.py  —  MobileNetV3-Small  +  FPN-lite  +  FCOS head
             Single class: person only.

FCOS (Fully Convolutional One-Stage Detector) is anchor-free.
Instead of predicting offsets from pre-defined anchor boxes,
each foreground pixel directly predicts:
    (l, t, r, b)  — distance to LEFT / TOP / RIGHT / BOTTOM edge of the box
    centerness    — how centred this pixel is within the person box
    cls_logit     — is this pixel inside a person?

This produces tighter boxes than SSD because every pixel votes
independently with no anchor shape constraint.

Architecture
────────────
Input  (B, 3, H, W)  — default 640 × 640

MobileNetV3-Small backbone
    block 8  →  C3  (B,  48, H/16, W/16)
    block 11 →  C4  (B,  96, H/32, W/32)
    block 12 →  C5  (B, 576, H/32, W/32)

FPN-lite neck  (top-down fusion → 128 ch uniform)
    P3 stride 16  — catches small / far persons
    P4 stride 32  — mid-size persons
    P5 stride 32  — kept from C5 lateral (large persons)
    P6 stride 64  — very large / close persons

FCOS head  (shared weights across all 4 FPN levels)
    4 depthwise-separable convs → cls branch + reg branch

Output per level  (B, ·, H_i, W_i):
    cls  (B,1,H,W)   raw logit
    reg  (B,4,H,W)   l t r b in pixels (positive, exp-activated)
    ctr  (B,1,H,W)   centerness logit
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision.models import MobileNet_V3_Small_Weights
from torchvision.ops import nms as tv_nms


# ─────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────

class DSConv(nn.Module):
    """Depthwise-separable conv → BN → ReLU."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch,  3, stride=stride,
                      padding=1, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class Conv1x1(nn.Module):
    """Pointwise conv → BN → ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


# ─────────────────────────────────────────────────────────────────
# FPN-lite neck
# ─────────────────────────────────────────────────────────────────

class FPNLite(nn.Module):
    """
    Lightweight top-down FPN.
    Fuses C3/C4/C5 from backbone into four 128-ch feature maps.
    """
    def __init__(self, in_channels=(48, 96, 576), out_ch=128):
        super().__init__()
        c3, c4, c5 = in_channels

        # lateral projections
        self.lat5 = Conv1x1(c5, out_ch)
        self.lat4 = Conv1x1(c4, out_ch)
        self.lat3 = Conv1x1(c3, out_ch)

        # post-merge refinement
        self.smooth4 = DSConv(out_ch, out_ch)
        self.smooth3 = DSConv(out_ch, out_ch)

        # extra coarse scale for very large persons
        self.p6_conv = nn.Sequential(
            nn.Conv2d(c5, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.smooth4(
            self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        )
        p3 = self.smooth3(
            self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        )
        p6 = self.p6_conv(c5)
        # strides: p3=16, p4=32, p5=32, p6=64
        return p3, p4, p5, p6


# ─────────────────────────────────────────────────────────────────
# FCOS head
# ─────────────────────────────────────────────────────────────────

class FCOSHead(nn.Module):
    """
    Shared FCOS head applied identically at each FPN level.

    Outputs per spatial location:
        cls  scalar  — is there a person centred here?
        reg  4-vec   — (l, t, r, b) distances in feature-map pixels
                       × stride = original-image pixels
        ctr  scalar  — centerness (how centred within the GT box)
    """
    def __init__(self, in_ch=128, num_convs=4):
        super().__init__()

        def _tower(n):
            layers = [DSConv(in_ch, in_ch)]
            for _ in range(n - 1):
                layers.append(DSConv(in_ch, in_ch))
            return nn.Sequential(*layers)

        self.cls_tower = _tower(num_convs)
        self.reg_tower = _tower(num_convs)

        self.cls_pred = nn.Conv2d(in_ch, 1, 1)  # person logit
        self.reg_pred = nn.Conv2d(in_ch, 4, 1)  # l t r b
        self.ctr_pred = nn.Conv2d(in_ch, 1, 1)  # centerness

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # prior prob = 0.01 → stable early training
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, feat, scale):
        """
        feat  : (B, C, H, W)
        scale : scalar nn.Parameter per FPN level
        """
        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)

        cls = self.cls_pred(cls_feat)                          # (B,1,H,W)
        ctr = self.ctr_pred(cls_feat)                          # (B,1,H,W)
        reg = torch.exp(scale * self.reg_pred(reg_feat))       # (B,4,H,W) >0
        return cls, reg, ctr


# ─────────────────────────────────────────────────────────────────
# PersonDetector
# ─────────────────────────────────────────────────────────────────

class PersonDetector(nn.Module):
    """
    MobileNetV3-Small  +  FPN-lite  +  FCOS  — person detector.

    forward() returns a list of 4 level-dicts:
        [{'cls':(B,1,H,W), 'reg':(B,4,H,W), 'ctr':(B,1,H,W), 'stride':int}, ...]

    decode() turns those into final xyxy boxes (normalised 0-1)
    for in-memory cropping by Model 2.
    """

    STRIDES = [16, 32, 32, 64]

    def __init__(self, pretrained=True, input_size=640):
        super().__init__()
        self.input_size = input_size

        # ── backbone ─────────────────────────────────────────────
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = tvm.mobilenet_v3_small(weights=weights).features

        # ── neck ─────────────────────────────────────────────────
        self.neck = FPNLite(in_channels=(48, 96, 576), out_ch=128)

        # ── shared head ──────────────────────────────────────────
        self.head = FCOSHead(in_ch=128, num_convs=4)

        # one learnable exp-scale per FPN level (keeps reg positive & stable)
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in self.STRIDES]
        )

    # ── backbone feature extraction ───────────────────────────────

    def _backbone(self, x):
        c3 = c4 = c5 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 8:  c3 = x   # (B,  48, H/16, W/16)
            if i == 11: c4 = x   # (B,  96, H/32, W/32)
            if i == 12: c5 = x   # (B, 576, H/32, W/32)
        return c3, c4, c5

    # ── forward ──────────────────────────────────────────────────

    def forward(self, x):
        c3, c4, c5     = self._backbone(x)
        p3, p4, p5, p6 = self.neck(c3, c4, c5)

        preds = []
        for feat, stride, scale in zip(
                [p3, p4, p5, p6], self.STRIDES, self.scales):
            cls, reg, ctr = self.head(feat, scale)
            preds.append({'cls': cls, 'reg': reg, 'ctr': ctr, 'stride': stride})
        return preds

    # ── decode (inference only) ───────────────────────────────────

    @torch.no_grad()
    def decode(self, preds, img_h, img_w,
               score_thresh=0.40, nms_iou=0.45):
        """
        Convert FCOS predictions → final detections.

        Returns list of dicts (one per image):
            boxes  (K,4)  xyxy normalised [0,1]
            scores (K,)
        """
        B      = preds[0]['cls'].size(0)
        device = preds[0]['cls'].device

        all_boxes  = [[] for _ in range(B)]
        all_scores = [[] for _ in range(B)]

        for level in preds:
            cls    = level['cls']     # (B,1,H,W)
            reg    = level['reg']     # (B,4,H,W)
            ctr    = level['ctr']     # (B,1,H,W)
            stride = level['stride']
            H, W   = cls.shape[-2:]

            # pixel-centre coordinates in original image space
            ys = (torch.arange(H, device=device).float() + 0.5) * stride
            xs = (torch.arange(W, device=device).float() + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # (H,W)
            cx = grid_x.reshape(-1)   # (H*W,)
            cy = grid_y.reshape(-1)

            # score = sqrt(cls_prob * centerness)  — FCOS standard
            score_map = (cls.sigmoid() * ctr.sigmoid()).sqrt()[:, 0]  # (B,H,W)

            for b in range(B):
                scores_flat = score_map[b].reshape(-1)       # (H*W,)
                reg_flat    = reg[b].permute(1,2,0).reshape(-1,4)  # (H*W,4)

                keep = scores_flat >= score_thresh
                if not keep.any():
                    continue

                s  = scores_flat[keep]
                r  = reg_flat[keep]          # l t r b  (pixels)
                px = cx[keep]
                py = cy[keep]

                x1 = (px - r[:,0]).clamp(0, img_w)
                y1 = (py - r[:,1]).clamp(0, img_h)
                x2 = (px + r[:,2]).clamp(0, img_w)
                y2 = (py + r[:,3]).clamp(0, img_h)

                valid = (x2 > x1 + 1) & (y2 > y1 + 1)
                if not valid.any():
                    continue

                all_boxes[b].append(torch.stack([x1,y1,x2,y2],1)[valid])
                all_scores[b].append(s[valid])

        results = []
        for b in range(B):
            if not all_boxes[b]:
                results.append({
                    'boxes':  torch.zeros(0,4,device=device),
                    'scores': torch.zeros(0,  device=device),
                })
                continue

            boxes  = torch.cat(all_boxes[b])
            scores = torch.cat(all_scores[b])

            keep   = tv_nms(boxes, scores, nms_iou)
            boxes  = boxes[keep]
            scores = scores[keep]

            # normalise to [0,1]
            boxes[:,[0,2]] /= img_w
            boxes[:,[1,3]] /= img_h
            boxes = boxes.clamp(0,1)

            results.append({'boxes': boxes, 'scores': scores})

        return results


# ─────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    m = PersonDetector(pretrained=False)
    m.eval()
    x = torch.zeros(2, 3, 640, 640)
    preds = m(x)
    print('FPN outputs:')
    for p in preds:
        print(f"  stride={p['stride']:2d}  cls={tuple(p['cls'].shape)}")
    dets = m.decode(preds, 640, 640, score_thresh=0.01)
    print(f'\nboxes[0]: {dets[0]["boxes"].shape}')
    print('model.py  OK')
