"""
utils/logger.py  —  KA-ResUNet++
Training curve visualization.
"""

import os
import csv
import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(log_csv: str, save_dir: str = "./results"):
    """
    Plot training and validation curves from CSV log.
    Generates 3 plots: Loss, Dice/IoU, HD95.
    """
    if not os.path.exists(log_csv):
        print(f"[Logger] Log file not found: {log_csv}")
        return

    df = pd.read_csv(log_csv)
    os.makedirs(save_dir, exist_ok=True)

    # ── Loss curve ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="steelblue")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="coral")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Dice / IoU ────────────────────────────────────────────────────────
    axes[1].plot(df["epoch"], df["train_dice"], label="Train Dice", color="steelblue", linestyle="--")
    axes[1].plot(df["epoch"], df["val_dice"],   label="Val Dice",   color="coral")
    if "val_iou" in df.columns:
        axes[1].plot(df["epoch"], df["val_iou"], label="Val IoU",   color="green")
    axes[1].set_title("Dice / IoU", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # ── HD95 ──────────────────────────────────────────────────────────────
    if "val_hd95" in df.columns:
        axes[2].plot(df["epoch"], df["val_hd95"], label="Val HD95", color="purple")
        axes[2].set_title("Hausdorff Distance 95 (lower is better)", fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("HD95")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    plt.suptitle("KA-ResUNet++ Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Logger] Training curves saved to: {out}")


def plot_ablation_table(results: dict, save_dir: str = "./results"):
    """
    Bar chart of Dice scores across ablation experiments.
    results: {'A1: Baseline': 0.880, 'A2: +KAN': 0.893, ...}
    """
    labels = list(results.keys())
    scores = list(results.values())
    colors = plt.cm.Blues([(s - min(scores)) / (max(scores) - min(scores) + 1e-5)
                            for s in scores])

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), scores, color=colors, edgecolor="navy", linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(min(scores) - 0.02, 1.0)
    ax.set_ylabel("Dice Score", fontsize=12)
    ax.set_title("Ablation Study — KA-ResUNet++", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{score:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(save_dir, "ablation_chart.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Logger] Ablation chart saved to: {out}")
