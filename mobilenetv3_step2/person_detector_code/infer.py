"""
infer.py  —  Inference wrapper for MobileNetV3 + FCOS PersonDetector.

Integration point between Model 1 (detector) and Model 2 (classifier).
Crops are returned as PIL Images in memory — no disk I/O.

Usage:
    from infer import PersonDetectorInference

    detector = PersonDetectorInference('runs/run1/best.pt')

    result = detector.detect_image(pil_image)

    for crop, box_px, score in zip(result['crops'],
                                   result['boxes_px'],
                                   result['scores']):
        label = model_2.predict(crop)   # PIL crop passed directly

CLI demo (draws boxes, saves annotated images):
    python infer.py runs/run1/best.pt image1.jpg image2.jpg
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from pathlib import Path

from model import PersonDetector

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _preprocess(images: list[Image.Image],
                input_size: int,
                device: torch.device) -> torch.Tensor:
    tensors = []
    for img in images:
        t = TF.to_tensor(TF.resize(img.convert('RGB'), [input_size, input_size]))
        t = TF.normalize(t, _MEAN, _STD)
        tensors.append(t)
    return torch.stack(tensors).to(device)


def _crop_person(image: Image.Image,
                 box_norm,        # (x1,y1,x2,y2) [0,1]
                 padding=0.05) -> Image.Image:
    """
    Crop a person from a PIL Image using normalised xyxy coords.
    Adds a small padding margin so Model 2 gets a little context.
    Crops at ORIGINAL image resolution, not 640px detector resolution.
    """
    W, H = image.size
    x1, y1, x2, y2 = box_norm
    bw, bh = x2 - x1, y2 - y1

    x1 = max(0.0, x1 - padding * bw)
    y1 = max(0.0, y1 - padding * bh)
    x2 = min(1.0, x2 + padding * bw)
    y2 = min(1.0, y2 + padding * bh)

    px1, py1 = int(x1 * W), int(y1 * H)
    px2, py2 = max(int(x2 * W), px1 + 1), max(int(y2 * H), py1 + 1)
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
    """

    def __init__(self,
                 checkpoint_path: str,
                 device: str | None = None,
                 score_thresh: float = 0.40,
                 nms_iou:      float = 0.45,
                 padding:      float = 0.05):

        self._device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        ckpt = torch.load(checkpoint_path, map_location=self._device)
        args = ckpt.get('args', {})

        self._input_size  = int(args.get('input_size', 640))
        self._score_thresh = score_thresh
        self._nms_iou     = nms_iou
        self._padding     = padding

        self._model = PersonDetector(
            pretrained=False, input_size=self._input_size)
        self._model.load_state_dict(ckpt['model'])
        self._model.eval().to(self._device)

        print(f'[PersonDetector-FCOS] loaded  device={self._device}  '
              f'input={self._input_size}px')

    # ── public API ────────────────────────────────────────────────

    def detect_image(self, image: Image.Image) -> dict:
        """
        Detect persons in a single PIL Image.

        Returns dict:
            crops      list[PIL.Image]          — one crop per person
            boxes_norm list[(x1,y1,x2,y2)]      — normalised [0,1]
            boxes_px   list[(x1,y1,x2,y2)]      — pixel coordinates
            scores     list[float]
        """
        return self.detect_batch([image])[0]

    def detect_batch(self, images: list[Image.Image]) -> list[dict]:
        """
        Detect persons in a batch of PIL Images.
        Returns list of dicts (one per image) — schema as detect_image().
        """
        tensor = _preprocess(images, self._input_size, self._device)

        with torch.no_grad():
            preds = self._model(tensor)
            dets  = self._model.decode(
                preds,
                img_h=self._input_size,
                img_w=self._input_size,
                score_thresh=self._score_thresh,
                nms_iou=self._nms_iou,
            )

        results = []
        for img, det in zip(images, dets):
            W, H     = img.size
            boxes_nm = det['boxes']    # (K,4) xyxy [0,1]
            scores   = det['scores']

            crops, boxes_px, norms, sc_list = [], [], [], []

            for i in range(len(scores)):
                nm = tuple(boxes_nm[i].tolist())       # (x1,y1,x2,y2) norm
                px = (int(nm[0]*W), int(nm[1]*H),
                      int(nm[2]*W), int(nm[3]*H))

                crop = _crop_person(img, nm, self._padding)

                crops.append(crop)
                boxes_px.append(px)
                norms.append(nm)
                sc_list.append(float(scores[i]))

            results.append({
                'crops':      crops,      # list[PIL.Image]  → Model 2
                'boxes_norm': norms,      # list[(x1,y1,x2,y2)] [0,1]
                'boxes_px':   boxes_px,   # list[(x1,y1,x2,y2)] pixels
                'scores':     sc_list,    # list[float]
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

    det = PersonDetectorInference(ckpt_path, score_thresh=0.35)
    images  = [Image.open(p).convert('RGB') for p in img_paths]
    results = det.detect_batch(images)

    for path, img, res in zip(img_paths, images, results):
        draw = ImageDraw.Draw(img)
        for (x1,y1,x2,y2), score in zip(res['boxes_px'], res['scores']):
            draw.rectangle([x1,y1,x2,y2], outline='red',   width=3)
            draw.text((x1+4, y1+4), f'{score:.2f}', fill='yellow')

        out_p = out_dir / (Path(path).stem + '_det.jpg')
        img.save(out_p)
        print(f'{path}  →  {len(res["scores"])} person(s)  →  {out_p}')
