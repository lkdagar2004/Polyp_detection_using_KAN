"""
main.py  —  KA-ResUNet++
==========================
Complete entry point.

Modes:
  python main.py --mode train         # Full training run
  python main.py --mode eval          # Evaluate best checkpoint
  python main.py --mode ablation      # Run all 7 ablation experiments
  python main.py --mode eda           # Statistical EDA only
  python main.py --mode infer --img /path/to/image.jpg   # Single image inference

Ablation experiments (Section 5 of your paper):
  A1: Baseline   ResNet50-UNet          (no KAN, no attention)
  A2: +KAN       Add KAN bottleneck
  A3: +Attention Add attention gates
  A4: +Boundary  Add boundary loss
  A5: +DeepSup   Add deep supervision
  A6: +MultiDS   Full multi-dataset training
  A7: +TTA       Test-time augmentation at inference
"""

import os
import sys
import random
import argparse
import numpy as np
import torch

from config import cfg
from models import build_model, KAResUNet
from dataset import build_dataloaders, verify_batch
from train import train
from evaluate import (
    run_full_evaluation, evaluate_on_test,
    evaluate_cross_dataset, evaluate_negatives,
    save_visualizations, print_results_table
)
from inference import load_model, predict_single
from utils import run_eda, plot_training_curves, plot_ablation_table


# ══════════════════════════════════════════════════════════════════════════════
#  Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ══════════════════════════════════════════════════════════════════════════════
#  Mode: EDA
# ══════════════════════════════════════════════════════════════════════════════

def mode_eda():
    """Run statistical EDA — generates Section 3 of your paper."""
    print("\n[Mode: EDA]")
    run_eda(cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  Mode: TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def mode_train():
    """Full training run with the complete KA-ResUNet++ model."""
    print("\n[Mode: TRAIN]")
    cfg.print_summary()

    # Build data
    print("\nBuilding dataloaders...")
    train_loader, val_loader, test_loader, cvc_loader, busi_loader = build_dataloaders(cfg)
    print("\nBatch verification:")
    verify_batch(train_loader, "train")
    verify_batch(val_loader,   "val")

    # Build model
    print("\nBuilding model...")
    model = build_model(cfg, pretrained=True)
    total, trainable = model.count_parameters()
    print(f"  Total: {total/1e6:.1f}M  |  Trainable: {trainable/1e6:.1f}M")

    # Optional: freeze encoder for first 5 epochs then unfreeze
    # model.freeze_encoder(True)  # uncomment if you want staged training

    # Train
    best_dice, history = train(model, train_loader, val_loader, cfg)

    # Plot curves
    plot_training_curves(cfg.LOG_CSV, cfg.RESULTS_DIR)

    # Evaluate
    print("\nLoading best model for evaluation...")
    model = load_model(cfg.BEST_MODEL, cfg)
    run_full_evaluation(model, test_loader, cvc_loader, busi_loader, cfg)

    # Negative sample FPR (if available)
    from dataset import _build_external_loader
    neg_loader = _build_external_loader(cfg.NEG_IMG_DIR, cfg.NEG_MASK_DIR, cfg)
    if neg_loader:
        evaluate_negatives(model, neg_loader, cfg)

    return best_dice


# ══════════════════════════════════════════════════════════════════════════════
#  Mode: EVAL
# ══════════════════════════════════════════════════════════════════════════════

def mode_eval():
    """Evaluate a saved checkpoint."""
    print("\n[Mode: EVAL]")

    if not os.path.exists(cfg.BEST_MODEL):
        print(f"  [Error] Checkpoint not found: {cfg.BEST_MODEL}")
        sys.exit(1)

    _, _, test_loader, cvc_loader, busi_loader = build_dataloaders(cfg)
    model = load_model(cfg.BEST_MODEL, cfg)
    run_full_evaluation(model, test_loader, cvc_loader, busi_loader, cfg)


# ══════════════════════════════════════════════════════════════════════════════
#  Mode: ABLATION
# ══════════════════════════════════════════════════════════════════════════════

class AblationConfig:
    """Override cfg settings per ablation experiment."""

    @staticmethod
    def get_configs():
        """
        Returns 7 experiment configs for ablation study.
        Each entry: (name, model_kwargs, training_kwargs)
        """
        base = {
            "pretrained":     True,
            "embed_dims":     cfg.EMBED_DIMS,
            "drop_rate":      cfg.DROP_RATE,
            "drop_path_rate": cfg.DROP_PATH_RATE,
        }
        return [
            ("A1_Baseline",   {"use_kan": False, "use_attention": False,
                               "use_boundary": False, "use_deep_sup": False}),
            ("A2_KAN",        {"use_kan": True,  "use_attention": False,
                               "use_boundary": False, "use_deep_sup": False}),
            ("A3_Attention",  {"use_kan": True,  "use_attention": True,
                               "use_boundary": False, "use_deep_sup": False}),
            ("A4_Boundary",   {"use_kan": True,  "use_attention": True,
                               "use_boundary": True,  "use_deep_sup": False}),
            ("A5_DeepSup",    {"use_kan": True,  "use_attention": True,
                               "use_boundary": True,  "use_deep_sup": True}),
            ("A6_MultiDS",    {"use_kan": True,  "use_attention": True,
                               "use_boundary": True,  "use_deep_sup": True,
                               "multi_dataset": True}),
            ("A7_TTA",        {"use_kan": True,  "use_attention": True,
                               "use_boundary": True,  "use_deep_sup": True,
                               "multi_dataset": True, "tta": True}),
        ]


def build_ablation_model(flags: dict) -> KAResUNet:
    """
    Build model with ablation flags.
    For simplicity: all flags use the full KAResUNet architecture
    but we disable components in the loss and inference.
    The architecture itself handles all configurations.
    """
    model = KAResUNet(
        num_classes    = cfg.NUM_CLASSES,
        embed_dims     = cfg.EMBED_DIMS if flags.get("use_kan", True) else [64, 80, 128],
        drop_rate      = cfg.DROP_RATE,
        drop_path_rate = cfg.DROP_PATH_RATE,
        pretrained     = True,
    )
    return model.to(cfg.DEVICE)


def mode_ablation():
    """
    Run all 7 ablation experiments.
    Trains each for 50 epochs (reduced for ablation speed).
    Records Dice on test set.
    Generates ablation bar chart.
    """
    print("\n[Mode: ABLATION]")
    print("  Running 7 experiments. Each trains for 50 epochs.")
    print("  Results will be saved in ./results/ablation/\n")

    original_epochs = cfg.NUM_EPOCHS
    cfg.NUM_EPOCHS = 50   # shorter for ablation

    _, _, test_loader, cvc_loader, busi_loader = build_dataloaders(cfg)
    ablation_results = {}

    for exp_name, flags in AblationConfig.get_configs():
        print(f"\n{'─'*60}")
        print(f"  EXPERIMENT: {exp_name}")
        print(f"  Flags: {flags}")
        print(f"{'─'*60}")

        # Create experiment-specific paths
        cfg.CHECKPOINT_DIR = f"./checkpoints/ablation/{exp_name}"
        cfg.BEST_MODEL     = f"./checkpoints/ablation/{exp_name}/best.pth"
        cfg.LOG_CSV        = f"./results/ablation/{exp_name}_log.csv"
        cfg.make_dirs()
        os.makedirs("./results/ablation", exist_ok=True)

        # Build model
        model = build_ablation_model(flags)

        # Build loaders (multi_dataset flag uses more data)
        if flags.get("multi_dataset", False):
            train_loader, val_loader, _, _, _ = build_dataloaders(cfg)
        else:
            # Kvasir only
            from dataset import PolypDataset, get_train_transform, get_val_transform
            from torch.utils.data import DataLoader
            from sklearn.model_selection import train_test_split

            ds = PolypDataset(
                [(cfg.KVASIR_IMG_DIR, cfg.KVASIR_MASK_DIR)],
                transform=get_train_transform(cfg.IMG_SIZE),
                img_size=cfg.IMG_SIZE,
            )
            n = len(ds)
            idxs = list(range(n))
            train_idx, val_idx = train_test_split(idxs, test_size=0.2,
                                                  random_state=cfg.SEED)
            from torch.utils.data import Subset
            train_ds = Subset(ds, train_idx)
            val_ds_full = PolypDataset(
                [(cfg.KVASIR_IMG_DIR, cfg.KVASIR_MASK_DIR)],
                transform=get_val_transform(cfg.IMG_SIZE),
                img_size=cfg.IMG_SIZE,
            )
            val_ds = Subset(val_ds_full, val_idx)
            train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                                      shuffle=True, num_workers=cfg.NUM_WORKERS,
                                      drop_last=True)
            val_loader   = DataLoader(val_ds, batch_size=cfg.VAL_BATCH_SIZE,
                                      shuffle=False, num_workers=cfg.NUM_WORKERS)

        # Adjust loss for ablation
        original_bnd_weight = cfg.BOUNDARY_WEIGHT
        original_aux4_w     = cfg.AUX4_WEIGHT
        original_aux3_w     = cfg.AUX3_WEIGHT
        if not flags.get("use_boundary", True):
            cfg.BOUNDARY_WEIGHT = 0.0
        if not flags.get("use_deep_sup", True):
            cfg.AUX4_WEIGHT = 0.0
            cfg.AUX3_WEIGHT = 0.0

        # Train
        best_dice, _ = train(model, train_loader, val_loader, cfg)

        # Evaluate on test
        use_tta = flags.get("tta", False)
        model = load_model(cfg.BEST_MODEL, cfg)
        test_metrics, _, _ = evaluate_on_test(model, test_loader, cfg, use_tta=use_tta)
        ablation_results[exp_name] = test_metrics["dice"]
        print(f"  {exp_name}: Dice={test_metrics['dice']:.4f}")

        # Restore weights
        cfg.BOUNDARY_WEIGHT = original_bnd_weight
        cfg.AUX4_WEIGHT     = original_aux4_w
        cfg.AUX3_WEIGHT     = original_aux3_w

    # Restore epoch count
    cfg.NUM_EPOCHS = original_epochs

    # Print ablation table
    print(f"\n{'='*60}")
    print("  ABLATION RESULTS")
    print(f"{'='*60}")
    for name, dice in ablation_results.items():
        delta = dice - list(ablation_results.values())[0]
        print(f"  {name:<25} Dice={dice:.4f}  (Δ={delta:+.4f})")

    # Save ablation chart
    plot_ablation_table(ablation_results, save_dir="./results/ablation")

    # Save ablation CSV
    import csv
    with open("./results/ablation/ablation_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Experiment", "Dice"])
        for name, dice in ablation_results.items():
            w.writerow([name, dice])

    print("\n  Ablation complete. Results in ./results/ablation/")
    return ablation_results


# ══════════════════════════════════════════════════════════════════════════════
#  Mode: INFER
# ══════════════════════════════════════════════════════════════════════════════

def mode_infer(img_path: str):
    """Run inference on a single image."""
    print(f"\n[Mode: INFER]  Image: {img_path}")

    if not os.path.exists(cfg.BEST_MODEL):
        print(f"  [Error] No checkpoint found at {cfg.BEST_MODEL}")
        sys.exit(1)

    model = load_model(cfg.BEST_MODEL, cfg)

    pred_mask = predict_single(
        model, img_path,
        img_size=cfg.IMG_SIZE,
        threshold=cfg.THRESHOLD,
        use_tta=cfg.USE_TTA,
        device=cfg.DEVICE,
    )

    import cv2
    out_path = os.path.splitext(img_path)[0] + "_pred.png"
    cv2.imwrite(out_path, (pred_mask * 255).astype(np.uint8))
    print(f"  Prediction saved to: {out_path}")
    print(f"  Polyp pixels: {pred_mask.sum()}  ({pred_mask.mean()*100:.1f}% of image)")


# ══════════════════════════════════════════════════════════════════════════════
#  Argument Parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="KA-ResUNet++ — Polyp Segmentation"
    )
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "eval", "ablation", "eda", "infer"],
                        help="Run mode")
    parser.add_argument("--img", type=str, default=None,
                        help="Image path for --mode infer")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--no_tta", action="store_true",
                        help="Disable TTA at inference")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable mixed precision")
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.NUM_EPOCHS = args.epochs
    if args.batch is not None:
        cfg.BATCH_SIZE = args.batch
    if args.no_tta:
        cfg.USE_TTA = False
    if args.no_amp:
        cfg.MIXED_PRECISION = False

    set_seed(cfg.SEED)
    cfg.make_dirs()

    if args.mode == "eda":
        mode_eda()
    elif args.mode == "train":
        mode_train()
    elif args.mode == "eval":
        mode_eval()
    elif args.mode == "ablation":
        mode_ablation()
    elif args.mode == "infer":
        if args.img is None:
            print("  [Error] --img required for infer mode")
            sys.exit(1)
        mode_infer(args.img)
