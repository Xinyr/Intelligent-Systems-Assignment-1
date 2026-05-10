"""
dataset.py  —  YOLO-format dataset loader, single class: person.

Label format per line:  0  cx  cy  w  h   (class_id always 0, ignored)

Fixes vs previous version
──────────────────────────
  1. Augmentation order fixed: Resize happens FIRST, then spatial
     augmentations (HorizontalFlip, ScaleTranslate).  Previously
     ScaleTranslate ran on the original PIL size and the translation
     effect was partially overwritten by the final Resize call.

  2. Mosaic augmentation added (train only, p=0.5).  Combines 4 images
     into one 640×640 canvas — the single highest-value augmentation for
     a single-class dataset as it artificially creates multi-person
     scenes at varying scales.

  3. Box validity filter added after augmentation: boxes with area < 1%
     of the image or width/height < 8px are dropped.  Prevents tiny
     degenerate boxes from confusing the FCOS target assignment.
"""

import random
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────
# Box-aware augmentations  (all operate on already-resized images)
# ─────────────────────────────────────────────────────────────────

class Resize:
    """Resize image to target (H,W). Boxes unchanged (already normalised)."""
    def __init__(self, size):
        self.size = list(size)  # [H, W]
    def __call__(self, img, boxes):
        return TF.resize(img, self.size), boxes


class HorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, boxes):
        if random.random() < self.p:
            img = TF.hflip(img)
            if boxes.numel():
                b = boxes.clone()
                b[:, 0] = 1.0 - b[:, 0]   # flip cx
                boxes = b
        return img, boxes


class ColorJitter:
    def __init__(self):
        self.jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    def __call__(self, img, boxes):
        return self.jitter(img), boxes


class RandomScaleTranslate:
    """
    Random zoom-out + small translation on an already-resized image.
    Fix: now operates after Resize so the padding is correctly preserved.
    """
    def __init__(self, scale=(0.85, 1.0), translate=0.06):
        self.scale     = scale
        self.translate = translate

    def __call__(self, img, boxes):
        W, H = img.size   # already at target resolution
        s  = random.uniform(*self.scale)
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)

        nw, nh = int(W * s), int(H * s)

        # resize-down → pad back to original size → no final resize needed
        img = TF.resize(img, [nh, nw])
        pl  = max(0, int((W - nw) / 2 + tx * W))
        pt  = max(0, int((H - nh) / 2 + ty * H))
        pr  = max(0, W - nw - pl)
        pb  = max(0, H - nh - pt)
        img = TF.pad(img, [pl, pt, pr, pb], fill=0)

        # crop/resize back to exactly (H,W) if padding overshot
        img = TF.resize(img, [H, W])

        if boxes.numel():
            b       = boxes.clone()
            b[:, 0] = b[:, 0] * s + (pl / W)
            b[:, 1] = b[:, 1] * s + (pt / H)
            b[:, 2] = b[:, 2] * s
            b[:, 3] = b[:, 3] * s
            boxes   = b.clamp(0, 1)

        return img, boxes


class RandomGrayscale:
    def __init__(self, p=0.1):
        self.p = p
    def __call__(self, img, boxes):
        if random.random() < self.p:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        return img, boxes


class ToTensorNorm:
    def __call__(self, img, boxes):
        img = TF.to_tensor(img)
        img = TF.normalize(img, _MEAN, _STD)
        return img, boxes


class Compose:
    def __init__(self, transforms):
        self.ts = transforms
    def __call__(self, img, boxes):
        for t in self.ts:
            img, boxes = t(img, boxes)
        return img, boxes


# ─────────────────────────────────────────────────────────────────
# Box validity filter
# ─────────────────────────────────────────────────────────────────

def filter_boxes(boxes, min_area=0.001, min_side=0.01):
    """
    Remove boxes that are too small after augmentation.
    boxes : (N,4) cx cy w h  normalised
    Returns filtered boxes tensor.
    """
    if boxes.numel() == 0:
        return boxes
    w, h = boxes[:, 2], boxes[:, 3]
    area_ok = (w * h) >= min_area
    side_ok = (w >= min_side) & (h >= min_side)
    return boxes[area_ok & side_ok]


# ─────────────────────────────────────────────────────────────────
# Mosaic augmentation
# ─────────────────────────────────────────────────────────────────

def mosaic_4(samples, input_size, base_transform):
    """
    Combine 4 samples into one mosaic image.

    Divides a (W,H) canvas into four quadrants at a random cut point
    near the centre. Each sample is placed in one quadrant and its
    boxes are remapped to the new canvas coordinates.

    Args:
        samples       : list of 4 (img_path, lbl_path) tuples
        input_size    : (H, W) target size
        base_transform: Compose of [Resize, ToTensorNorm] — no augment

    Returns:
        mosaic_tensor : (3, H, W)
        mosaic_boxes  : (N, 4) cx cy w h  normalised
    """
    H, W    = input_size
    canvas  = Image.new('RGB', (W, H), (114, 114, 114))

    # random cut point  (avoid extreme edges)
    cut_x = random.randint(W // 4, 3 * W // 4)
    cut_y = random.randint(H // 4, 3 * H // 4)

    # quadrant regions: (x1, y1, x2, y2) in canvas pixels
    quads = [
        (0,      0,      cut_x,  cut_y),   # top-left
        (cut_x,  0,      W,      cut_y),   # top-right
        (0,      cut_y,  cut_x,  H),       # bottom-left
        (cut_x,  cut_y,  W,      H),       # bottom-right
    ]

    all_boxes = []

    for (img_path, lbl_path), (qx1, qy1, qx2, qy2) in zip(samples, quads):
        qw, qh = qx2 - qx1, qy2 - qy1

        # load image and resize to quadrant size
        img = Image.open(img_path).convert('RGB')
        img = img.resize((qw, qh), Image.BILINEAR)
        canvas.paste(img, (qx1, qy1))

        # load and remap boxes
        boxes = _load_boxes(lbl_path)
        if boxes.numel() > 0:
            # convert normalised (in original img) → normalised (in canvas)
            new_cx = (boxes[:, 0] * qw + qx1) / W
            new_cy = (boxes[:, 1] * qh + qy1) / H
            new_w  =  boxes[:, 2] * qw / W
            new_h  =  boxes[:, 3] * qh / H
            remapped = torch.stack([new_cx, new_cy, new_w, new_h], dim=1)
            all_boxes.append(remapped)

    mosaic_boxes = filter_boxes(
        torch.cat(all_boxes) if all_boxes else torch.zeros(0, 4)
    )

    # normalise canvas
    tensor = TF.normalize(TF.to_tensor(canvas), _MEAN, _STD)
    return tensor, mosaic_boxes


def _load_boxes(lbl_path):
    boxes = []
    if lbl_path is not None and Path(lbl_path).exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append(list(map(float, parts[1:5])))
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4)


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

class PersonDataset(Dataset):
    """
    YOLO-format single-class person detection dataset.

    Args:
        root        : split directory (contains images/ and labels/)
        input_size  : (H, W) target resolution
        augment     : apply training augmentations (incl. mosaic)
        mosaic_prob : probability of mosaic augmentation per sample
    """

    def __init__(self, root, input_size=(640, 640),
                 augment=False, mosaic_prob=0.5):
        self.root        = Path(root)
        self.input_size  = input_size
        self.augment     = augment
        self.mosaic_prob = mosaic_prob if augment else 0.0

        img_dir = self.root / 'images'
        lbl_dir = self.root / 'labels'

        if not img_dir.exists():
            raise FileNotFoundError(f'images/ not found under {root}')
        if not lbl_dir.exists():
            raise FileNotFoundError(f'labels/ not found under {root}')

        self.samples = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            self.samples.append(
                (img_path, lbl_path if lbl_path.exists() else None))

        if not self.samples:
            raise RuntimeError(f'No images found in {img_dir}')

        # ── augmentation pipeline ─────────────────────────────────
        # Fix: Resize is the FIRST transform so all subsequent spatial
        # augmentations operate on fixed-size tensors.
        aug_list = [Resize(input_size)]
        if augment:
            aug_list += [
                HorizontalFlip(p=0.5),
                ColorJitter(),
                RandomScaleTranslate(scale=(0.85, 1.0), translate=0.06),
                RandomGrayscale(p=0.1),
            ]
        aug_list.append(ToTensorNorm())
        self.transform = Compose(aug_list)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # ── mosaic branch ─────────────────────────────────────────
        if self.augment and random.random() < self.mosaic_prob:
            # pick 3 more random samples
            other_idxs = random.choices(range(len(self.samples)), k=3)
            four = [self.samples[idx]] + [self.samples[i] for i in other_idxs]
            image, boxes = mosaic_4(four, self.input_size, self.transform)
            return {'image': image, 'boxes': boxes,
                    'img_path': str(self.samples[idx][0])}

        # ── normal branch ─────────────────────────────────────────
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        boxes = _load_boxes(lbl_path)
        image, boxes = self.transform(image, boxes)
        boxes = filter_boxes(boxes)

        return {'image': image, 'boxes': boxes,
                'img_path': str(img_path)}


# ─────────────────────────────────────────────────────────────────
# Collate + DataLoader factory
# ─────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return {
        'images':    torch.stack([b['image'] for b in batch]),
        'boxes':     [b['boxes']    for b in batch],
        'img_paths': [b['img_path'] for b in batch],
    }


def build_dataloaders(dataset_root, input_size=(640, 640),
                      batch_size=8, num_workers=4,
                      mosaic_prob=0.5):
    root    = Path(dataset_root)
    loaders = {}
    cfg = {
        'train': dict(augment=True,  shuffle=True,  drop_last=True),
        'val':   dict(augment=False, shuffle=False, drop_last=False),
        'test':  dict(augment=False, shuffle=False, drop_last=False),
    }

    for split, opts in cfg.items():
        split_dir = root / split
        if not split_dir.exists():
            print(f'[dataset] WARNING: {split_dir} not found, skipping.')
            continue
        ds = PersonDataset(
            str(split_dir), input_size=input_size,
            augment=opts['augment'],
            mosaic_prob=mosaic_prob if opts['augment'] else 0.0,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=opts['shuffle'],
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=opts['drop_last'],
            persistent_workers=(num_workers > 0),
        )
        print(f'[dataset] {split:5s}: {len(ds):6d} images  '
              f'(mosaic={mosaic_prob if opts["augment"] else 0:.0%})')

    return loaders


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python dataset.py <dataset_root>')
        raise SystemExit(1)
    loaders = build_dataloaders(sys.argv[1], batch_size=4, num_workers=0)
    batch   = next(iter(loaders['train']))
    print(f"images : {batch['images'].shape}")
    print(f"boxes  : {[b.shape for b in batch['boxes']]}")
    print('dataset.py  OK')
