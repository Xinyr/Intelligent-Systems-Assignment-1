"""
model.py  —  MobileNetV3-Small  +  FPN-lite  +  FCOS head
             Single class: person only.

FCOS is anchor-free. Each foreground pixel directly predicts:
    (l, t, r, b)  — distance to LEFT/TOP/RIGHT/BOTTOM edge of the box
    centerness    — how centred this pixel is within the person box
    cls_logit     — is this pixel inside a person?

Architecture
────────────
Input  (B, 3, H, W)  — default 640 × 640

MobileNetV3-Small backbone
    block 8  →  C3  (B,  48, H/16, W/16)
    block 11 →  C4  (B,  96, H/32, W/32)
    block 12 →  C5  (B, 576, H/32, W/32)

FPN-lite neck  (all four output strides are distinct)
    P3  stride  16  — small / far persons
    P4  stride  32  — medium persons
    P5  stride  64  — large persons
    P6  stride 128  — very large / close persons

FCOS head  (shared weights across all 4 FPN levels)
    4 depthwise-separable convs (BN+ReLU after EACH sub-conv)
    → cls / reg / centerness branches

Fixes vs previous version
──────────────────────────
  1. DSConv: BN+ReLU now applied after depthwise AND pointwise
             (previously only after pointwise — unbounded activations)
  2. FPN: P5 pooled to stride-64 so P4 and P5 are at different resolutions
          (previously both at stride-32 — duplicate detection level)
  3. STRIDES = [16, 32, 64, 128]  — no duplicates
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
    """
    Depthwise-separable conv.
    BN + ReLU applied after EACH sub-conv (depthwise and pointwise).
    Previously BN was only after pointwise — depthwise had no normalisation.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # depthwise
        self.dw     = nn.Conv2d(in_ch, in_ch, 3, stride=stride,
                                padding=1, groups=in_ch, bias=False)
        self.dw_bn  = nn.BatchNorm2d(in_ch)
        self.dw_act = nn.ReLU(inplace=True)
        # pointwise
        self.pw     = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.pw_bn  = nn.BatchNorm2d(out_ch)
        self.pw_act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw_act(self.dw_bn(self.dw(x)))
        x = self.pw_act(self.pw_bn(self.pw(x)))
        return x


class Conv1x1(nn.Module):
    """Pointwise conv → BN → ReLU  (FPN lateral projections)."""
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
    Lightweight top-down FPN.  All four output strides are distinct:
        P3 → stride  16   (from C3 + top-down from P4)
        P4 → stride  32   (from C4 + top-down from lat5)
        P5 → stride  64   (lat5 downsampled by 2)
        P6 → stride 128   (C5 downsampled by 4)

    Fix: previously P4 and P5 were both at H/32 (stride 32) because
    C4 and C5 share that spatial size.  P5 is now explicitly pooled
    to H/64 so each FPN level covers a unique scale.
    """
    def __init__(self, in_channels=(48, 96, 576), out_ch=128):
        super().__init__()
        c3, c4, c5 = in_channels

        self.lat5 = Conv1x1(c5, out_ch)   # C5 lateral  → 128ch @ H/32
        self.lat4 = Conv1x1(c4, out_ch)   # C4 lateral  → 128ch @ H/32
        self.lat3 = Conv1x1(c3, out_ch)   # C3 lateral  → 128ch @ H/16

        self.smooth4 = DSConv(out_ch, out_ch)  # refine P4
        self.smooth3 = DSConv(out_ch, out_ch)  # refine P3

        # P5: stride-64  (learnable stride-2 downsample of lat5 output)
        self.p5_down = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=2,
                      padding=1, groups=out_ch, bias=False),   # dw
            nn.Conv2d(out_ch, out_ch, 1, bias=False),           # pw
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # P6: stride-128  (two stride-2 convs from C5)
        self.p6_conv = nn.Sequential(
            nn.Conv2d(c5, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, c3, c4, c5):
        lat5 = self.lat5(c5)                                              # H/32
        p4   = self.smooth4(
            self.lat4(c4) + F.interpolate(lat5, size=c4.shape[-2:],
                                          mode='nearest'))                # H/32
        p3   = self.smooth3(
            self.lat3(c3) + F.interpolate(p4,   size=c3.shape[-2:],
                                          mode='nearest'))                # H/16
        p5   = self.p5_down(lat5)                                         # H/64
        p6   = self.p6_conv(c5)                                           # H/128
        return p3, p4, p5, p6   # strides: 16, 32, 64, 128  ✓


# ─────────────────────────────────────────────────────────────────
# FCOS head
# ─────────────────────────────────────────────────────────────────

class FCOSHead(nn.Module):
    """Shared FCOS head applied at each FPN level."""

    def __init__(self, in_ch=128, num_convs=4):
        super().__init__()

        def _tower(n):
            layers = [DSConv(in_ch, in_ch)]
            for _ in range(n - 1):
                layers.append(DSConv(in_ch, in_ch))
            return nn.Sequential(*layers)

        self.cls_tower = _tower(num_convs)
        self.reg_tower = _tower(num_convs)

        self.cls_pred = nn.Conv2d(in_ch, 1, 1)   # person logit
        self.reg_pred = nn.Conv2d(in_ch, 4, 1)   # l t r b
        self.ctr_pred = nn.Conv2d(in_ch, 1, 1)   # centerness logit

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # prior prob = 0.01 → avoids exploding focal loss at epoch 0
        nn.init.constant_(self.cls_pred.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, feat, scale):
        cls_feat = self.cls_tower(feat)
        reg_feat = self.reg_tower(feat)
        cls = self.cls_pred(cls_feat)                     # (B,1,H,W)
        ctr = self.ctr_pred(cls_feat)                     # (B,1,H,W)
        reg = torch.exp(scale * self.reg_pred(reg_feat))  # (B,4,H,W) >0
        return cls, reg, ctr


# ─────────────────────────────────────────────────────────────────
# PersonDetector
# ─────────────────────────────────────────────────────────────────

class PersonDetector(nn.Module):
    """
    MobileNetV3-Small  +  FPN-lite  +  FCOS  — single-class person detector.

    forward() → list of 4 level-dicts:
        [{'cls':(B,1,H,W), 'reg':(B,4,H,W), 'ctr':(B,1,H,W), 'stride':int}, ...]

    decode() → list of dicts (one per image):
        {'boxes':(K,4) xyxy normalised [0,1], 'scores':(K,)}
    """

    STRIDES = [16, 32, 64, 128]   # all distinct, no duplicate stride-32

    def __init__(self, pretrained=True, input_size=640):
        super().__init__()
        self.input_size = input_size

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = tvm.mobilenet_v3_small(weights=weights).features

        self.neck   = FPNLite(in_channels=(48, 96, 576), out_ch=128)
        self.head   = FCOSHead(in_ch=128, num_convs=4)
        self.scales = nn.ParameterList(
            [nn.Parameter(torch.ones(1)) for _ in self.STRIDES])

    def _backbone(self, x):
        c3 = c4 = c5 = None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 8:  c3 = x   # (B,  48, H/16, W/16)
            if i == 11: c4 = x   # (B,  96, H/32, W/32)
            if i == 12: c5 = x   # (B, 576, H/32, W/32)
        return c3, c4, c5

    def forward(self, x):
        c3, c4, c5     = self._backbone(x)
        p3, p4, p5, p6 = self.neck(c3, c4, c5)

        preds = []
        for feat, stride, scale in zip(
                [p3, p4, p5, p6], self.STRIDES, self.scales):
            cls, reg, ctr = self.head(feat, scale)
            preds.append({'cls': cls, 'reg': reg,
                          'ctr': ctr, 'stride': stride})
        return preds

    @torch.no_grad()
    def decode(self, preds, img_h, img_w,
               score_thresh=0.40, nms_iou=0.45):
        """
        Convert raw FCOS predictions → final detections.

        Returns list of dicts (one per image):
            boxes  (K,4)  xyxy normalised [0,1]
            scores (K,)
        """
        B      = preds[0]['cls'].size(0)
        device = preds[0]['cls'].device

        all_boxes  = [[] for _ in range(B)]
        all_scores = [[] for _ in range(B)]

        for level in preds:
            cls, reg, ctr = level['cls'], level['reg'], level['ctr']
            stride = level['stride']
            H, W   = cls.shape[-2:]

            ys = (torch.arange(H, device=device).float() + 0.5) * stride
            xs = (torch.arange(W, device=device).float() + 0.5) * stride
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            cx = grid_x.reshape(-1)
            cy = grid_y.reshape(-1)

            score_map = (cls.sigmoid() * ctr.sigmoid()).sqrt()[:, 0]

            for b in range(B):
                s_flat = score_map[b].reshape(-1)
                r_flat = reg[b].permute(1, 2, 0).reshape(-1, 4)

                keep = s_flat >= score_thresh
                if not keep.any():
                    continue

                s  = s_flat[keep]
                r  = r_flat[keep]
                px = cx[keep]
                py = cy[keep]

                x1 = (px - r[:, 0]).clamp(0, img_w)
                y1 = (py - r[:, 1]).clamp(0, img_h)
                x2 = (px + r[:, 2]).clamp(0, img_w)
                y2 = (py + r[:, 3]).clamp(0, img_h)

                valid = (x2 > x1 + 1) & (y2 > y1 + 1)
                if not valid.any():
                    continue

                all_boxes[b].append(torch.stack([x1, y1, x2, y2], 1)[valid])
                all_scores[b].append(s[valid])

        results = []
        for b in range(B):
            if not all_boxes[b]:
                results.append({'boxes':  torch.zeros(0, 4, device=device),
                                'scores': torch.zeros(0,    device=device)})
                continue

            boxes  = torch.cat(all_boxes[b])
            scores = torch.cat(all_scores[b])
            keep   = tv_nms(boxes, scores, nms_iou)
            boxes, scores = boxes[keep], scores[keep]

            boxes[:, [0, 2]] /= img_w
            boxes[:, [1, 3]] /= img_h
            results.append({'boxes': boxes.clamp(0, 1), 'scores': scores})

        return results


if __name__ == '__main__':
    m = PersonDetector(pretrained=False)
    m.eval()
    x = torch.zeros(2, 3, 640, 640)
    preds = m(x)
    print('FPN outputs:')
    for p in preds:
        print(f"  stride={p['stride']:3d}  cls={tuple(p['cls'].shape)}")
    dets = m.decode(preds, 640, 640, score_thresh=0.01)
    print(f'\nboxes[0]: {dets[0]["boxes"].shape}')
    print('model.py  OK')
