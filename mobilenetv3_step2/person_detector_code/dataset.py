"""
dataset.py  —  YOLO-format dataset loader.
               Single class: person (class_id always 0, ignored).

Expected layout:
    dataset/
        train/  images/  *.jpg|png    labels/  *.txt
        val/    images/               labels/
        test/   images/               labels/

Label format (one line per person):
    0  cx  cy  w  h      (all normalised 0-1, class_id=0 ignored)
"""

import random
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ─────────────────────────────────────────────────────────────────
# Box-aware augmentations
# ─────────────────────────────────────────────────────────────────

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
    """Random zoom-out + small translation. Keeps boxes inside canvas."""
    def __init__(self, scale=(0.8, 1.0), translate=0.08):
        self.scale     = scale
        self.translate = translate

    def __call__(self, img, boxes):
        W, H = img.size
        s  = random.uniform(*self.scale)
        tx = random.uniform(-self.translate, self.translate)
        ty = random.uniform(-self.translate, self.translate)

        nw, nh = int(W * s), int(H * s)
        img = TF.resize(img, [nh, nw])

        pl = int(tx * W + (W - nw) / 2)
        pt = int(ty * H + (H - nh) / 2)
        pr = W - nw - pl
        pb = H - nh - pt
        pl, pt = max(0, pl), max(0, pt)
        pr, pb = max(0, pr), max(0, pb)
        img = TF.pad(img, [pl, pt, pr, pb], fill=0)
        img = TF.resize(img, [H, W])

        if boxes.numel():
            b       = boxes.clone()
            b[:, 0] = b[:, 0] * s + tx + (1 - s) / 2
            b[:, 1] = b[:, 1] * s + ty + (1 - s) / 2
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


class Resize:
    def __init__(self, size):   # size = (H, W)
        self.size = size
    def __call__(self, img, boxes):
        return TF.resize(img, list(self.size)), boxes


class ToTensorNorm:
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    def __call__(self, img, boxes):
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.MEAN, self.STD)
        return img, boxes


class Compose:
    def __init__(self, transforms):
        self.ts = transforms
    def __call__(self, img, boxes):
        for t in self.ts:
            img, boxes = t(img, boxes)
        return img, boxes


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────

class PersonDataset(Dataset):
    """
    YOLO-format single-class person detection dataset.

    Args:
        root       : split directory (contains images/ and labels/)
        input_size : (H, W) to resize all images to
        augment    : apply training augmentations
    """

    def __init__(self, root, input_size=(640, 640), augment=False):
        self.root       = Path(root)
        self.input_size = input_size

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
            self.samples.append((img_path, lbl_path if lbl_path.exists() else None))

        if not self.samples:
            raise RuntimeError(f'No images found in {img_dir}')

        aug = []
        if augment:
            aug += [
                HorizontalFlip(p=0.5),
                ColorJitter(),
                RandomScaleTranslate(scale=(0.85, 1.0), translate=0.06),
                RandomGrayscale(p=0.1),
            ]
        aug += [Resize(input_size), ToTensorNorm()]
        self.transform = Compose(aug)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        boxes = []
        if lbl_path is not None:
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    # class_id cx cy w h — class_id ignored (always person)
                    cx, cy, bw, bh = map(float, parts[1:5])
                    boxes.append([cx, cy, bw, bh])

        boxes = torch.tensor(boxes, dtype=torch.float32)  # (N,4) or (0,4)
        image, boxes = self.transform(image, boxes)

        return {
            'image':    image,           # (3,H,W)
            'boxes':    boxes,           # (N,4)  cx cy w h  [0,1]
            'img_path': str(img_path),
        }


# ─────────────────────────────────────────────────────────────────
# Collate + DataLoader factory
# ─────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return {
        'images':    torch.stack([b['image'] for b in batch]),
        'boxes':     [b['boxes']    for b in batch],
        'img_paths': [b['img_path'] for b in batch],
    }


def build_dataloaders(dataset_root, input_size=(640,640),
                      batch_size=8, num_workers=4):
    """
    Build train/val/test DataLoaders from a YOLO-split dataset root.

    Returns dict with keys 'train', 'val', 'test' (whichever exist).
    """
    root    = Path(dataset_root)
    loaders = {}
    cfg     = {
        'train': dict(augment=True,  shuffle=True,  drop_last=True),
        'val':   dict(augment=False, shuffle=False, drop_last=False),
        'test':  dict(augment=False, shuffle=False, drop_last=False),
    }

    for split, opts in cfg.items():
        split_dir = root / split
        if not split_dir.exists():
            print(f'[dataset] WARNING: {split_dir} not found, skipping.')
            continue
        ds = PersonDataset(str(split_dir), input_size=input_size,
                           augment=opts['augment'])
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=opts['shuffle'],
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=opts['drop_last'],
        )
        print(f'[dataset] {split:5s}: {len(ds):6d} images')

    return loaders


# ─────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────

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
