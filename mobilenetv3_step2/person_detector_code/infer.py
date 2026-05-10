"""
infer.py  —  Inference wrapper for MobileNetV3 + FCOS PersonDetector.

Integration point between Model 1 (detector) and Model 2 (classifier).
Crops are returned as PIL Images in memory — no disk I/O.

Improvements vs previous version
──────────────────────────────────
  1. Aspect-ratio preserving resize with letterboxing for preprocessing.
     Squashing non-square images to 640×640 distorts person proportions
     and hurts detection.  Letterboxing pads with grey and rescales the
     decoded boxes back to the original coordinate frame correctly.

  2. Batch guard: detect_batch() auto-chunks large lists so GPU OOM
     is avoided when the caller passes hundreds of images at once.

  3. decode() called with original image dimensions (img_h, img_w) not
     input_size, so box coordinates are correct before normalisation.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from pathlib import Path

from model import PersonDetector

_MEAN       = [0.485, 0.456, 0.406]
_STD        = [0.229, 0.224, 0.225]
_PAD_VALUE  = 114   # grey letterbox fill (same as YOLO convention)


# ─────────────────────────────────────────────────────────────────
# Letterbox preprocessing
# ─────────────────────────────────────────────────────────────────

def _letterbox(img: Image.Image, target: int):
    """
    Resize image to (target × target) while preserving aspect ratio.
    Pads the shorter axis with grey.

    Returns:
        tensor    : (3, target, target) normalised
        scale     : float  — resize scale applied to original
        pad_left  : int    — pixels padded on the left
        pad_top   : int    — pixels padded on top
    """
    W, H    = img.size
    scale   = target / max(W, H)
    new_w   = int(round(W * scale))
    new_h   = int(round(H * scale))

    img_r   = img.resize((new_w, new_h), Image.BILINEAR)

    # centre the resized image on a grey canvas
    canvas  = Image.new('RGB', (target, target), (_PAD_VALUE,) * 3)
    pad_left = (target - new_w) // 2
    pad_top  = (target - new_h) // 2
    canvas.paste(img_r, (pad_left, pad_top))

    tensor  = TF.normalize(TF.to_tensor(canvas), _MEAN, _STD)
    return tensor, scale, pad_left, pad_top


def _unletterbox_boxes(boxes_norm, target, scale, pad_left, pad_top,
                       orig_w, orig_h):
    """
    Convert boxes from letterboxed-normalised space back to
    original-image-normalised space.

    boxes_norm : (K,4) xyxy in [0,1] relative to the (target,target) canvas
    Returns    : (K,4) xyxy in [0,1] relative to the original image
    """
    if boxes_norm.numel() == 0:
        return boxes_norm

    # canvas pixels
    b = boxes_norm.clone()
    b[:, [0, 2]] *= target
    b[:, [1, 3]] *= target

    # remove padding
    b[:, [0, 2]] -= pad_left
    b[:, [1, 3]] -= pad_top

    # reverse resize scale
    b /= scale

    # normalise to original image size
    b[:, [0, 2]] /= orig_w
    b[:, [1, 3]] /= orig_h

    return b.clamp(0, 1)


# ─────────────────────────────────────────────────────────────────
# Crop helper
# ─────────────────────────────────────────────────────────────────

def _crop_person(image: Image.Image, box_norm, padding=0.05):
    """
    Crop person from original PIL Image using normalised xyxy coords.
    Adds padding margin.  Crops at ORIGINAL resolution.
    """
    W, H = image.size
    x1, y1, x2, y2 = box_norm
    bw, bh = x2 - x1, y2 - y1

    x1 = max(0.0, x1 - padding * bw)
    y1 = max(0.0, y1 - padding * bh)
    x2 = min(1.0, x2 + padding * bw)
    y2 = min(1.0, y2 + padding * bh)

    px1 = int(x1 * W)
    py1 = int(y1 * H)
    px2 = max(int(x2 * W), px1 + 1)
    py2 = max(int(y2 * H), py1 + 1)
    return image.crop((px1, py1, px2, py2))


# ─────────────────────────────────────────────────────────────────
# Inference wrapper
# ─────────────────────────────────────────────────────────────────

class PersonDetectorInference:
    """
    Load-once inference wrapper for the FCOS PersonDetector.

    Args:
        checkpoint_path : path to best.pt / last.pt
        device          : 'cuda', 'cpu', or None (auto)
        score_thresh    : minimum detection score to keep
        nms_iou         : NMS IoU threshold
        padding         : fractional margin added around each crop
        max_batch       : max images per GPU forward pass (OOM guard)
    """

    def __init__(self,
                 checkpoint_path: str,
                 device: str | None  = None,
                 score_thresh: float = 0.40,
                 nms_iou:      float = 0.45,
                 padding:      float = 0.05,
                 max_batch:    int   = 16):

        self._device = torch.device(
            device if device
            else ('cuda' if torch.cuda.is_available() else 'cpu'))

        ckpt = torch.load(checkpoint_path,
                          map_location=self._device,
                          weights_only=False)
        saved_args = ckpt.get('args', {})

        self._input_size   = int(saved_args.get('input_size', 640))
        self._score_thresh = score_thresh
        self._nms_iou      = nms_iou
        self._padding      = padding
        self._max_batch    = max_batch

        self._model = PersonDetector(
            pretrained=False, input_size=self._input_size)
        self._model.load_state_dict(ckpt['model'])
        self._model.eval().to(self._device)

        print(f'[PersonDetector-FCOS] loaded  '
              f'device={self._device}  '
              f'input={self._input_size}px')

    # ── public API ────────────────────────────────────────────────

    def detect_image(self, image: Image.Image) -> dict:
        """
        Detect persons in a single PIL Image.

        Returns dict:
            crops      list[PIL.Image]       — one crop per person
            boxes_norm list[(x1,y1,x2,y2)]  — normalised [0,1] original img
            boxes_px   list[(x1,y1,x2,y2)]  — pixel coords in original img
            scores     list[float]
        """
        return self.detect_batch([image])[0]

    def detect_batch(self, images: list[Image.Image]) -> list[dict]:
        """
        Detect persons in a batch of PIL Images.
        Auto-chunks into max_batch-size sub-batches to avoid GPU OOM.
        """
        all_results = []
        for i in range(0, len(images), self._max_batch):
            chunk = images[i: i + self._max_batch]
            all_results.extend(self._forward_chunk(chunk))
        return all_results

    def _forward_chunk(self, images: list[Image.Image]) -> list[dict]:
        target    = self._input_size
        tensors   = []
        meta      = []   # (scale, pad_left, pad_top, orig_w, orig_h)

        for img in images:
            img_rgb = img.convert('RGB')
            t, scale, pl, pt = _letterbox(img_rgb, target)
            tensors.append(t)
            meta.append((scale, pl, pt, img_rgb.width, img_rgb.height))

        batch = torch.stack(tensors).to(self._device)

        with torch.no_grad():
            preds = self._model(batch)
            # decode in letterbox space (target × target)
            dets  = self._model.decode(
                preds,
                img_h=target,
                img_w=target,
                score_thresh=self._score_thresh,
                nms_iou=self._nms_iou,
            )

        results = []
        for img, det, (scale, pl, pt, orig_w, orig_h) in \
                zip(images, dets, meta):

            boxes_lb = det['boxes']    # (K,4) xyxy in letterbox-norm
            scores   = det['scores']

            # convert boxes back to original image coordinates
            boxes_orig = _unletterbox_boxes(
                boxes_lb, target, scale, pl, pt, orig_w, orig_h)

            crops, boxes_px, norms, sc_list = [], [], [], []
            for i in range(len(scores)):
                nm = tuple(boxes_orig[i].tolist())
                px = (int(nm[0] * orig_w), int(nm[1] * orig_h),
                      int(nm[2] * orig_w), int(nm[3] * orig_h))
                crop = _crop_person(img.convert('RGB'), nm, self._padding)
                crops.append(crop)
                boxes_px.append(px)
                norms.append(nm)
                sc_list.append(float(scores[i]))

            results.append({
                'crops':      crops,      # list[PIL.Image]  → Model 2
                'boxes_norm': norms,
                'boxes_px':   boxes_px,
                'scores':     sc_list,
            })

        return results


# ─────────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('Usage: python infer.py <checkpoint.pt> <img> [img ...]')
        sys.exit(1)

    ckpt_path = sys.argv[1]
    img_paths = sys.argv[2:]
    out_dir   = Path('infer_output')
    out_dir.mkdir(exist_ok=True)

    det     = PersonDetectorInference(ckpt_path, score_thresh=0.35)
    images  = [Image.open(p).convert('RGB') for p in img_paths]
    results = det.detect_batch(images)

    for path, img, res in zip(img_paths, images, results):
        draw = ImageDraw.Draw(img)
        for (x1, y1, x2, y2), score in zip(res['boxes_px'], res['scores']):
            draw.rectangle([x1, y1, x2, y2], outline='red',   width=3)
            draw.text((x1 + 4, y1 + 4), f'{score:.2f}', fill='yellow')
        out_p = out_dir / (Path(path).stem + '_det.jpg')
        img.save(out_p)
        print(f'{path}  →  {len(res["scores"])} person(s)  →  {out_p}')
