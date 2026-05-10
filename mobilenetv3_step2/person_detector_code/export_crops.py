"""
export_crops.py  —  Run trained PersonDetector over a dataset and save
                    one cropped image per detected person to disk.

This produces the input dataset for Model 2 (classifier).

Output:
    crops_output/
        train/   img001_person00.jpg  ...
        val/
        test/
        train_manifest.csv
        val_manifest.csv
        test_manifest.csv

Usage:
    python export_crops.py \\
        --checkpoint runs/run1/best.pt \\
        --data_root  /path/to/dataset \\
        --output_dir crops_output
"""

import csv, argparse
from pathlib import Path
from PIL import Image
import torch

from infer import PersonDetectorInference

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',   required=True)
    p.add_argument('--data_root',    required=True)
    p.add_argument('--output_dir',   default='crops_output')
    p.add_argument('--splits',       nargs='+',
                   default=['train', 'val', 'test'])
    p.add_argument('--score_thresh', type=float, default=0.35)
    p.add_argument('--nms_iou',      type=float, default=0.45)
    p.add_argument('--padding',      type=float, default=0.05)
    p.add_argument('--min_box_area', type=float, default=0.005,
                   help='Skip boxes < this fraction of image area')
    p.add_argument('--min_side_px',  type=int,   default=20,
                   help='Skip crops narrower or shorter than this many pixels')
    p.add_argument('--save_format',  default='jpg',
                   choices=['jpg', 'png'])
    p.add_argument('--max_batch',    type=int,   default=8,
                   help='Images per GPU forward pass')
    return p.parse_args()


def export_split(detector, split_dir, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(
        p for p in (split_dir / 'images').iterdir()
        if p.suffix.lower() in IMAGE_EXTS)

    csv_rows = []
    n_crops = n_skip = 0

    for img_path in img_paths:
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'  [skip] {img_path.name}: {e}')
            continue

        W, H   = image.size
        result = detector.detect_image(image)

        if not result['crops']:
            n_skip += 1
            continue

        saved = 0
        for i, (crop, bpx, bnm, score) in enumerate(zip(
                result['crops'], result['boxes_px'],
                result['boxes_norm'], result['scores'])):

            x1, y1, x2, y2 = bpx

            # area filter
            if ((x2 - x1) * (y2 - y1)) / max(W * H, 1) < args.min_box_area:
                continue
            # minimum pixel size filter
            if (x2 - x1) < args.min_side_px or (y2 - y1) < args.min_side_px:
                continue

            fname    = f'{img_path.stem}_person{i:02d}.{args.save_format}'
            out_path = out_dir / fname
            if args.save_format == 'jpg':
                crop.save(out_path, 'JPEG', quality=95)
            else:
                crop.save(out_path, 'PNG')

            csv_rows.append(dict(
                crop_path=str(out_path),
                source_img=str(img_path),
                x1_norm=round(bnm[0], 6), y1_norm=round(bnm[1], 6),
                x2_norm=round(bnm[2], 6), y2_norm=round(bnm[3], 6),
                x1_px=x1, y1_px=y1, x2_px=x2, y2_px=y2,
                score=round(score, 4),
                crop_w=crop.width, crop_h=crop.height,
            ))
            n_crops += 1
            saved   += 1

        if saved == 0:
            n_skip += 1

        if n_crops % 500 == 0 and n_crops > 0:
            print(f'    {n_crops} crops so far…')

    return n_crops, n_skip, csv_rows


def main():
    args = parse_args()

    device   = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = PersonDetectorInference(
        args.checkpoint, device=device,
        score_thresh=args.score_thresh,
        nms_iou=args.nms_iou,
        padding=args.padding,
        max_batch=args.max_batch,
    )

    data_root   = Path(args.data_root)
    out_root    = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    grand_crops = 0

    for split in args.splits:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f'[skip] {split}/ not found')
            continue

        print(f'\n── {split} ──')
        n_crops, n_skip, rows = export_split(
            detector, split_dir, out_root / split, args)

        if rows:
            csv_path = out_root / f'{split}_manifest.csv'
            with open(csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            print(f'  manifest → {csv_path}')

        print(f'  {n_crops} crops saved  '
              f'({n_skip} images with 0 valid detections)')
        grand_crops += n_crops

    print(f'\n✅  Total crops: {grand_crops}  →  {out_root.resolve()}')


if __name__ == '__main__':
    main()
