"""
train.py  —  KA-ResUNet++
===========================
Full training + validation loop.

Sources:
  AdamW setup            → Code 4, Cell 51 (with differential LR added)
  AverageMeter           → Code 2, Cell 10
  CSV logging            → Code 2's OrderedDict pattern
  clip_grad_norm_        → Code 4, Cell 51
  Model checkpointing    → Code 4, Cell 51 (save on best val Dice)

NEW:
  Differential LR        → KAN fc layers get cfg.KAN_LR < CNN layers cfg.LR
  CosineAnnealingWarmRestarts → upgrade from ReduceLROnPlateau
  Mixed precision        → torch.cuda.amp.autocast + GradScaler
  Early stopping         → on val Dice (not loss)
"""

import os
import csv
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from metrics import (
    MetricsTracker, compute_all_metrics,
    dice_score, iou_score
)
from losses import CombinedLoss


# ══════════════════════════════════════════════════════════════════════════════
#  Build optimizer with differential LR
# ══════════════════════════════════════════════════════════════════════════════

def build_optimizer(model, cfg):
    """
    Differential learning rates:
      - KAN fc layers (spline + base weights) → cfg.KAN_LR  (lower)
      - All other parameters                  → cfg.LR
    """
    kan_params   = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # KANLinear layers contain 'base_weight', 'spline_weight', 'spline_scaler'
        if any(k in name for k in ['base_weight', 'spline_weight', 'spline_scaler', 'kan_b']):
            kan_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': kan_params,   'lr': cfg.KAN_LR,  'weight_decay': 0.0},
        {'params': other_params, 'lr': cfg.LR,       'weight_decay': cfg.WEIGHT_DECAY},
    ]
    print(f"  [Optimizer] KAN params: {sum(p.numel() for p in kan_params):,} "
          f"| Other params: {sum(p.numel() for p in other_params):,}")

    return torch.optim.AdamW(param_groups)


def build_scheduler(optimizer, cfg):
    """CosineAnnealingWarmRestarts — better than ReduceLROnPlateau for KAN."""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.SCHEDULER_T0,
        T_mult=cfg.SCHEDULER_T_MULT,
        eta_min=cfg.SCHEDULER_ETA_MIN,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Train one epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler,
                    device, epoch, cfg, use_amp=True):
    """
    Run one full training epoch.

    Returns:
        OrderedDict with epoch metrics for logging
    """
    model.train()

    from metrics import AverageMeter
    loss_meter  = AverageMeter()
    bce_meter   = AverageMeter()
    dice_meter  = AverageMeter()
    bnd_meter   = AverageMeter()
    train_dice  = AverageMeter()
    train_iou   = AverageMeter()

    t_start = time.time()

    for batch_idx, (imgs, seg_gt, bnd_gt) in enumerate(loader):
        imgs   = imgs.to(device,   non_blocking=True)
        seg_gt = seg_gt.to(device, non_blocking=True)
        bnd_gt = bnd_gt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward (with optional mixed precision) ────────────────────────
        if use_amp and device != "cpu":
            with autocast():
                seg_logits, bnd_logits, aux4, aux3 = model(imgs)
                loss, loss_dict = criterion(
                    seg_logits, bnd_logits, aux4, aux3, seg_gt, bnd_gt
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            seg_logits, bnd_logits, aux4, aux3 = model(imgs)
            loss, loss_dict = criterion(
                seg_logits, bnd_logits, aux4, aux3, seg_gt, bnd_gt
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        # ── Cosine LR step (per-batch) ─────────────────────────────────────
        scheduler.step(epoch + batch_idx / len(loader))

        # ── Metrics ───────────────────────────────────────────────────────
        n = imgs.size(0)
        loss_meter.update(loss_dict["total"], n)
        bce_meter.update(loss_dict["bce_seg"], n)
        dice_meter.update(loss_dict["dice"], n)
        bnd_meter.update(loss_dict["bnd"], n)

        with torch.no_grad():
            preds = torch.sigmoid(seg_logits).detach().cpu()
            gts   = seg_gt.detach().cpu()
            for i in range(preds.size(0)):
                train_dice.update(dice_score(preds[i], gts[i], cfg.THRESHOLD))
                train_iou.update(iou_score(preds[i],  gts[i], cfg.THRESHOLD))

        # ── Console progress ───────────────────────────────────────────────
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(loader):
            elapsed = time.time() - t_start
            lr_now  = optimizer.param_groups[1]['lr']
            print(f"  Epoch[{epoch+1}] [{batch_idx+1}/{len(loader)}]  "
                  f"loss={loss_meter.avg:.4f}  "
                  f"dice={train_dice.avg:.4f}  "
                  f"lr={lr_now:.6f}  "
                  f"[{elapsed:.0f}s]")

    return OrderedDict([
        ("train_loss",  round(loss_meter.avg,  5)),
        ("train_bce",   round(bce_meter.avg,   5)),
        ("train_dice_loss", round(dice_meter.avg, 5)),
        ("train_bnd",   round(bnd_meter.avg,   5)),
        ("train_dice",  round(train_dice.avg,  5)),
        ("train_iou",   round(train_iou.avg,   5)),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  Validate one epoch
# ══════════════════════════════════════════════════════════════════════════════

def validate_one_epoch(model, loader, criterion, device, cfg):
    """
    Run validation.

    Returns:
        OrderedDict with all validation metrics
    """
    model.eval()

    from metrics import AverageMeter
    loss_meter  = AverageMeter()
    tracker     = MetricsTracker()

    with torch.no_grad():
        for imgs, seg_gt, bnd_gt in loader:
            imgs   = imgs.to(device,   non_blocking=True)
            seg_gt = seg_gt.to(device, non_blocking=True)
            bnd_gt = bnd_gt.to(device, non_blocking=True)

            seg_logits, bnd_logits, aux4, aux3 = model(imgs)
            loss, _ = criterion(seg_logits, bnd_logits, aux4, aux3, seg_gt, bnd_gt)
            loss_meter.update(loss.item(), imgs.size(0))

            preds = torch.sigmoid(seg_logits).cpu()
            gts   = seg_gt.cpu()

            for i in range(preds.size(0)):
                m = compute_all_metrics(preds[i], gts[i], cfg.THRESHOLD)
                tracker.update(m, n=1)

    avgs = tracker.get_averages()
    return OrderedDict([
        ("val_loss",        round(loss_meter.avg,        5)),
        ("val_dice",        round(avgs["dice"],          5)),
        ("val_iou",         round(avgs["iou"],           5)),
        ("val_precision",   round(avgs["precision"],     5)),
        ("val_recall",      round(avgs["recall"],        5)),
        ("val_specificity", round(avgs["specificity"],   5)),
        ("val_f1",          round(avgs["f1"],            5)),
        ("val_hd95",        round(avgs["hd95"],          3)),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  CSV Logger
# ══════════════════════════════════════════════════════════════════════════════

class CSVLogger:
    """Logs training history to CSV file (Code 2's OrderedDict pattern)."""
    def __init__(self, path: str):
        self.path  = path
        self.keys  = None
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, metrics: dict, epoch: int, lr: float):
        row = {"epoch": epoch + 1, "lr": round(lr, 8), **metrics}
        if self.keys is None:
            self.keys = list(row.keys())
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self.keys).writeheader()
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.keys).writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
#  EarlyStopping
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Stop training when val Dice stops improving."""
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_dice = -1.0

    def __call__(self, val_dice: float) -> bool:
        if val_dice > self.best_dice + self.min_delta:
            self.best_dice = val_dice
            self.counter   = 0
            return False   # don't stop
        else:
            self.counter += 1
            return self.counter >= self.patience


# ══════════════════════════════════════════════════════════════════════════════
#  Main training function
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_loader, val_loader, cfg):
    """
    Full training loop with:
      - Mixed precision
      - Differential LR (KAN vs CNN)
      - CosineAnnealingWarmRestarts
      - Gradient clipping
      - Best model checkpointing on val Dice
      - Early stopping on val Dice
      - CSV logging of all metrics

    Returns:
        best_val_dice : float
        history       : list of dicts
    """
    cfg.make_dirs()
    device = cfg.DEVICE

    # Build training components
    criterion  = build_criterion_from_cfg(cfg)
    optimizer  = build_optimizer(model, cfg)
    scheduler  = build_scheduler(optimizer, cfg)
    scaler     = GradScaler(enabled=(cfg.MIXED_PRECISION and device != "cpu"))
    stopper    = EarlyStopping(patience=cfg.EARLY_STOP_PATIENCE)
    logger     = CSVLogger(cfg.LOG_CSV)

    best_val_dice = -1.0
    history = []

    print(f"\n{'='*60}")
    print(f"  Training {cfg.PROJECT_NAME}")
    total_p, train_p = model.count_parameters()
    print(f"  Total params: {total_p/1e6:.1f}M | Trainable: {train_p/1e6:.1f}M")
    print(f"  Epochs: {cfg.NUM_EPOCHS} | Patience: {cfg.EARLY_STOP_PATIENCE}")
    print(f"{'='*60}\n")

    for epoch in range(cfg.NUM_EPOCHS):
        epoch_start = time.time()
        lr_now = optimizer.param_groups[1]["lr"]

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            scheduler, device, epoch, cfg,
            use_amp=cfg.MIXED_PRECISION
        )

        # Validate
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device, cfg
        )

        epoch_time = time.time() - epoch_start
        val_dice   = val_metrics["val_dice"]

        # ── Console summary ────────────────────────────────────────────────
        print(f"\nEpoch [{epoch+1:3d}/{cfg.NUM_EPOCHS}]  "
              f"time={epoch_time:.0f}s  lr={lr_now:.6f}")
        print(f"  Train: loss={train_metrics['train_loss']:.4f}  "
              f"dice={train_metrics['train_dice']:.4f}  "
              f"iou={train_metrics['train_iou']:.4f}")
        print(f"  Val:   loss={val_metrics['val_loss']:.4f}  "
              f"dice={val_metrics['val_dice']:.4f}  "
              f"iou={val_metrics['val_iou']:.4f}  "
              f"hd95={val_metrics['val_hd95']:.2f}  "
              f"spec={val_metrics['val_specificity']:.4f}")

        # ── Checkpoint (save on best val Dice) ─────────────────────────────
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                "epoch":         epoch + 1,
                "model_state":   model.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_dice": best_val_dice,
                "val_metrics":   val_metrics,
                "cfg": {
                    "IMG_SIZE":    cfg.IMG_SIZE,
                    "EMBED_DIMS":  cfg.EMBED_DIMS,
                    "NUM_CLASSES": cfg.NUM_CLASSES,
                }
            }, cfg.BEST_MODEL)
            print(f"  ✓ Saved best model  (val_dice={best_val_dice:.4f})")

        # ── Log to CSV ─────────────────────────────────────────────────────
        all_metrics = {**train_metrics, **val_metrics}
        logger.write(all_metrics, epoch, lr_now)
        history.append({"epoch": epoch + 1, **all_metrics})

        # ── Early stopping ─────────────────────────────────────────────────
        if stopper(val_dice):
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {cfg.EARLY_STOP_PATIENCE} epochs)")
            break

        print()

    print(f"\n{'='*60}")
    print(f"  Training complete.  Best val Dice: {best_val_dice:.4f}")
    print(f"  Best model saved to: {cfg.BEST_MODEL}")
    print(f"  Log saved to: {cfg.LOG_CSV}")
    print(f"{'='*60}\n")

    return best_val_dice, history


# ── Import helper so train.py can standalone ────────────────────────────────

def build_criterion_from_cfg(cfg):
    from losses import build_criterion
    return build_criterion(cfg)
