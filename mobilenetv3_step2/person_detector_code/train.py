"""
train.py  —  Training script for MobileNetV3 + FCOS person detector.

Usage:
    python train.py --data_root /path/to/dataset --epochs 80 --batch 8

Improvements vs previous version
──────────────────────────────────
  1. LR warmup (5 epochs linear from 1e-5 → LR) stabilises the
     randomly-initialised head before the backbone starts adapting.

  2. Backbone frozen for the first FREEZE_EPOCHS epochs so the head
     learns sensible outputs before the pretrained weights move.

  3. CosineAnnealingWarmRestarts (T_0=20) replaces the single cosine
     decay — multiple restarts give the model chances to escape local
     minima, typically +2–4 mAP on small datasets.

  4. Early stopping with PATIENCE + WARMUP_SAVE_EPOCH to prevent
     checkpoint-selection bias inflating val mAP.  No checkpoint is
     saved before WARMUP_SAVE_EPOCH so lucky early spikes are ignored.

  5. Threshold sweep at the end runs on VAL, not test — the best
     threshold is then used for a single clean test evaluation.
"""

import os, sys, time, argparse, json
from pathlib import Path

import torch
import torch.optim as optim
from torchvision.ops import box_iou as tv_box_iou

from model   import PersonDetector
from dataset import build_dataloaders
from loss    import FCOSLoss


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',         required=True)
    p.add_argument('--epochs',            type=int,   default=80)
    p.add_argument('--batch',             type=int,   default=8)
    p.add_argument('--input_size',        type=int,   default=640)
    p.add_argument('--lr',                type=float, default=1e-3)
    p.add_argument('--weight_decay',      type=float, default=1e-4)
    p.add_argument('--workers',           type=int,   default=4)
    p.add_argument('--run_name',          type=str,   default='run1')
    p.add_argument('--resume',            type=str,   default=None)
    p.add_argument('--score_thresh',      type=float, default=0.35)
    p.add_argument('--nms_iou',           type=float, default=0.45)
    p.add_argument('--no_amp',            action='store_true')
    p.add_argument('--warmup_epochs',     type=int,   default=5,
                   help='Linear LR warmup epochs')
    p.add_argument('--freeze_epochs',     type=int,   default=5,
                   help='Freeze backbone for first N epochs')
    p.add_argument('--patience',          type=int,   default=15,
                   help='Early-stop patience (epochs without improvement)')
    p.add_argument('--warmup_save_epoch', type=int,   default=10,
                   help='Do not save best.pt before this epoch')
    p.add_argument('--mosaic_prob',       type=float, default=0.5)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# mAP@0.5
# ─────────────────────────────────────────────────────────────────

def compute_map50(model, loader, device, score_thresh, nms_iou):
    model.eval()
    all_preds  = []
    n_gt_total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            H, W   = images.shape[-2:]
            preds  = model(images)
            dets   = model.decode(preds, H, W,
                                  score_thresh=score_thresh,
                                  nms_iou=nms_iou)

            for det, gt_boxes in zip(dets, batch['boxes']):
                gt_boxes = gt_boxes.to(device)
                n_gt_total += gt_boxes.size(0)

                pred_boxes  = det['boxes']
                pred_scores = det['scores']

                if pred_boxes.numel() == 0:
                    continue
                if gt_boxes.numel() == 0:
                    for s in pred_scores:
                        all_preds.append((s.item(), 0))
                    continue

                gt_xyxy = torch.stack([
                    gt_boxes[:, 0] - gt_boxes[:, 2] / 2,
                    gt_boxes[:, 1] - gt_boxes[:, 3] / 2,
                    gt_boxes[:, 0] + gt_boxes[:, 2] / 2,
                    gt_boxes[:, 1] + gt_boxes[:, 3] / 2,
                ], dim=1)

                iou        = tv_box_iou(pred_boxes, gt_xyxy)
                matched_gt = set()
                for ki in pred_scores.argsort(descending=True):
                    bv, bj = iou[ki].max(dim=0)
                    if bv.item() >= 0.5 and bj.item() not in matched_gt:
                        all_preds.append((pred_scores[ki].item(), 1))
                        matched_gt.add(bj.item())
                    else:
                        all_preds.append((pred_scores[ki].item(), 0))

    if n_gt_total == 0 or not all_preds:
        return 0.0

    all_preds.sort(key=lambda x: -x[0])
    tp   = torch.tensor([x[1] for x in all_preds], dtype=torch.float32).cumsum(0)
    fp   = (1 - torch.tensor([x[1] for x in all_preds],
                              dtype=torch.float32)).cumsum(0)
    prec = tp / (tp + fp).clamp(min=1e-6)
    rec  = tp / max(n_gt_total, 1)

    return sum(
        prec[rec >= t].max().item() if (rec >= t).any() else 0.0
        for t in torch.linspace(0, 1, 101)
    ) / 101.0


# ─────────────────────────────────────────────────────────────────
# LR warmup helper
# ─────────────────────────────────────────────────────────────────

def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
    """Linear warmup: epoch 1 → LR/warmup_epochs … epoch N → LR."""
    if epoch <= warmup_epochs:
        factor = epoch / warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * factor


# ─────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler,
                device, use_amp, epoch, input_size):
    model.train()
    sum_loss = sum_cls = sum_reg = sum_ctr = 0.0
    t0 = time.time()

    for step, batch in enumerate(loader):
        images   = batch['images'].to(device, non_blocking=True)
        gt_boxes = batch['boxes']

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type,
                                enabled=use_amp):
            preds = model(images)
            loss, cls_l, reg_l, ctr_l = criterion(
                preds, gt_boxes, input_size, input_size)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        sum_loss += loss.item()
        sum_cls  += cls_l.item()
        sum_reg  += reg_l.item()
        sum_ctr  += ctr_l.item()

        if (step + 1) % 50 == 0 or (step + 1) == len(loader):
            print(f'  [E{epoch:03d} {step+1:4d}/{len(loader)}] '
                  f'loss={sum_loss/(step+1):.4f}  '
                  f'cls={sum_cls/(step+1):.4f}  '
                  f'reg={sum_reg/(step+1):.4f}  '
                  f'ctr={sum_ctr/(step+1):.4f}  '
                  f'({time.time()-t0:.0f}s)')

    n = len(loader)
    return sum_loss/n, sum_cls/n, sum_reg/n, sum_ctr/n


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'
    print(f'Device: {device}  AMP: {use_amp}')

    run_dir   = Path('runs') / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = run_dir / 'best.pt'
    ckpt_last = run_dir / 'last.pt'

    with open(run_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── data ───────────────────────────────────────────────────
    input_size = (args.input_size, args.input_size)
    loaders    = build_dataloaders(
        args.data_root, input_size=input_size,
        batch_size=args.batch, num_workers=args.workers,
        mosaic_prob=args.mosaic_prob)

    # ── model ──────────────────────────────────────────────────
    model = PersonDetector(pretrained=True,
                           input_size=args.input_size).to(device)

    # ── param groups (backbone 10× lower LR) ───────────────────
    backbone_p = list(model.features.parameters())
    head_p     = (list(model.neck.parameters()) +
                  list(model.head.parameters()) +
                  list(model.scales))

    optimizer = optim.AdamW([
        {'params': backbone_p, 'lr': args.lr / 10,
         'initial_lr': args.lr / 10},
        {'params': head_p,     'lr': args.lr,
         'initial_lr': args.lr},
    ], weight_decay=args.weight_decay)

    # CosineAnnealingWarmRestarts — T_0 = 20 epochs per restart
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6)

    scaler    = torch.amp.GradScaler(enabled=use_amp)
    criterion = FCOSLoss(cls_weight=1.0, reg_weight=1.5, ctr_weight=1.0)

    # ── resume ─────────────────────────────────────────────────
    start_epoch      = 1
    best_map50       = 0.0
    epochs_no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch       = ckpt['epoch'] + 1
        best_map50        = ckpt.get('best_map50', 0.0)
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        print(f'Resumed epoch {ckpt["epoch"]}  '
              f'best={best_map50:.4f}  '
              f'no_improve={epochs_no_improve}')

    # ── training loop ───────────────────────────────────────────
    log = []
    for epoch in range(start_epoch, args.epochs + 1):

        # ── backbone freeze phase ─────────────────────────────
        if epoch <= args.freeze_epochs:
            for p in model.features.parameters():
                p.requires_grad_(False)
            if epoch == 1:
                print(f'Backbone frozen for epochs 1–{args.freeze_epochs}')
        elif epoch == args.freeze_epochs + 1:
            for p in model.features.parameters():
                p.requires_grad_(True)
            print(f'Backbone unfrozen at epoch {epoch}')

        # ── LR warmup ─────────────────────────────────────────
        if epoch <= args.warmup_epochs:
            warmup_lr(optimizer, epoch, args.warmup_epochs, args.lr)

        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{args.epochs}   '
              f'lr_head={optimizer.param_groups[1]["lr"]:.2e}  '
              f'lr_backbone={optimizer.param_groups[0]["lr"]:.2e}')
        print(f'{"="*60}')

        tr_loss, tr_cls, tr_reg, tr_ctr = train_epoch(
            model, loaders['train'], criterion, optimizer,
            scaler, device, use_amp, epoch, args.input_size)

        # step scheduler only after warmup
        if epoch > args.warmup_epochs:
            scheduler.step(epoch - args.warmup_epochs)

        # ── validation mAP ────────────────────────────────────
        val_ap50 = 0.0
        if 'val' in loaders:
            val_ap50 = compute_map50(model, loaders['val'], device,
                                     args.score_thresh, args.nms_iou)
            print(f'  Val mAP@0.5 = {val_ap50:.4f}')

        # ── checkpoint ────────────────────────────────────────
        state = dict(
            epoch=epoch,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            best_map50=best_map50,
            epochs_no_improve=epochs_no_improve,
            args=vars(args),
        )
        torch.save(state, ckpt_last)

        # only consider saving best after warmup window
        if epoch >= args.warmup_save_epoch and val_ap50 > best_map50:
            best_map50        = val_ap50
            epochs_no_improve = 0
            state['best_map50'] = best_map50
            torch.save(state, ckpt_best)
            print(f'  *** New best mAP@0.5 = {best_map50:.4f} → best.pt')
        elif epoch >= args.warmup_save_epoch:
            epochs_no_improve += 1

        log.append(dict(epoch=epoch,
                        loss=round(tr_loss, 5), cls=round(tr_cls, 5),
                        reg=round(tr_reg, 5),   ctr=round(tr_ctr, 5),
                        val_ap50=round(val_ap50, 5),
                        no_improve=epochs_no_improve))
        with open(run_dir / 'log.json', 'w') as f:
            json.dump(log, f, indent=2)

        # ── early stopping ────────────────────────────────────
        if (epoch >= args.warmup_save_epoch and
                epochs_no_improve >= args.patience):
            print(f'\nEarly stop — no improvement for '
                  f'{args.patience} epochs.')
            break

    # ── final test evaluation (val threshold sweep first) ──────
    print('\n' + '='*60)
    if 'test' in loaders and ckpt_best.exists():
        ckpt = torch.load(ckpt_best, map_location=device)
        model.load_state_dict(ckpt['model'])

        # sweep on VAL to pick best threshold
        print('Threshold sweep on val set...')
        best_thresh = args.score_thresh
        best_f1     = 0.0
        for thresh in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
            ap = compute_map50(model, loaders['val'], device,
                               thresh, args.nms_iou)
            if ap > best_f1:
                best_f1     = ap
                best_thresh = thresh
        print(f'Best val threshold: {best_thresh:.2f}  (mAP={best_f1:.4f})')

        # single clean test evaluation
        test_ap50 = compute_map50(model, loaders['test'], device,
                                  best_thresh, args.nms_iou)
        print(f'Test mAP@0.5 = {test_ap50:.4f}  '
              f'(threshold={best_thresh})')

    print(f'Weights → {run_dir}')


if __name__ == '__main__':
    main()
