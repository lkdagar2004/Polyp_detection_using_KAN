"""
evaluate.py  —  KA-ResUNet++
==============================
Complete evaluation suite producing all results needed for paper tables.

Sources:
  evaluate_on_test   → Code 1 (Cell 17) timing + Code 2 (Cell 14) validate
  cross_dataset      → NEW (your cross-dataset novelty contribution)
  size_stratified    → NEW (from Code 4 EDA coverage bins)
  negative_fpr       → NEW (clinical FPR contribution)
  save_visualizations → Code 1 (show_predictions pattern)

Produces:
  - Main results table (9 metrics)
  - Ablation-ready single number per experiment
  - Cross-dataset table
  - Per-size Dice table
  - Negative sample FPR
  - Visualization grid (image | GT | prediction | boundary)
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional

from metrics import (
    MetricsTracker, compute_all_metrics,
    compute_size_stratified_metrics,
    fpr_on_negatives, InferenceTimer
)
from inference import predict_tta


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate_on_loader  —  core eval function
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_on_loader(model, loader, cfg, use_tta=False, loader_name="test"):
    """
    Evaluate model on any DataLoader.
    Returns full metrics dict and collected preds/gts for further analysis.
    """
    model.eval()
    device  = cfg.DEVICE
    tracker = MetricsTracker()
    timer   = InferenceTimer()

    all_preds = []
    all_gts   = []

    for imgs, seg_gt, bnd_gt in loader:
        imgs   = imgs.to(device, non_blocking=True)
        seg_gt = seg_gt.to(device, non_blocking=True)

        timer.start()
        if use_tta:
            prob = predict_tta(model, imgs, cfg.THRESHOLD)
        else:
            seg_logits, _, _, _ = model(imgs)
            prob = torch.sigmoid(seg_logits)
        timer.stop()

        preds = prob.cpu()
        gts   = seg_gt.cpu()

        all_preds.extend([preds[i] for i in range(preds.size(0))])
        all_gts.extend([gts[i] for i in range(gts.size(0))])

        for i in range(preds.size(0)):
            m = compute_all_metrics(preds[i], gts[i], cfg.THRESHOLD)
            tracker.update(m, n=1)

    avgs = tracker.get_averages()
    avgs["inference_ms"] = timer.mean_ms()

    print(f"\n[{loader_name}] Results:")
    print(f"  Dice={avgs['dice']:.4f}  IoU={avgs['iou']:.4f}  "
          f"Precision={avgs['precision']:.4f}  Recall={avgs['recall']:.4f}")
    print(f"  Specificity={avgs['specificity']:.4f}  F1={avgs['f1']:.4f}  "
          f"HD95={avgs['hd95']:.2f}  Infer={avgs['inference_ms']:.1f}ms/img")

    return avgs, all_preds, all_gts


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate_on_test  —  primary evaluation on held-out test set
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_on_test(model, test_loader, cfg, use_tta=True):
    """Primary evaluation on held-out Kvasir-SEG test set."""
    print(f"\n{'='*60}")
    print(f"  PRIMARY EVALUATION — Kvasir-SEG Test Set")
    print(f"  TTA: {use_tta}")
    print(f"{'='*60}")

    metrics, preds, gts = evaluate_on_loader(
        model, test_loader, cfg, use_tta=use_tta, loader_name="Kvasir-SEG test"
    )
    return metrics, preds, gts


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate_cross_dataset  —  zero-shot cross-dataset evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_cross_dataset(model, cvc_loader, busi_loader, cfg):
    """
    Zero-shot cross-dataset evaluation.
    Model was trained on Kvasir-SEG — test on CVC and BUSI without retraining.
    This is your generalizability contribution.
    """
    print(f"\n{'='*60}")
    print(f"  CROSS-DATASET EVALUATION (zero-shot generalization)")
    print(f"{'='*60}")

    results = {}
    if cvc_loader is not None:
        cvc_metrics, _, _ = evaluate_on_loader(
            model, cvc_loader, cfg, use_tta=True, loader_name="CVC-ClinicDB (zero-shot)"
        )
        results["CVC-ClinicDB"] = cvc_metrics
    else:
        print("  [Warning] CVC-ClinicDB loader not available.")

    if busi_loader is not None:
        busi_metrics, _, _ = evaluate_on_loader(
            model, busi_loader, cfg, use_tta=True, loader_name="BUSI (zero-shot)"
        )
        results["BUSI"] = busi_metrics
    else:
        print("  [Warning] BUSI loader not available.")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate_by_size  —  size-stratified metrics (Code 4 EDA bins)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_by_size(preds, gts, cfg):
    """
    Compute Dice broken down by polyp coverage size.
    Uses Code 4's 5-bin scheme: empty / small / medium / large / huge.
    This is a novel per-size analysis for your paper.
    """
    print(f"\n{'='*60}")
    print(f"  SIZE-STRATIFIED EVALUATION")
    print(f"{'='*60}")

    results = compute_size_stratified_metrics(preds, gts, cfg.THRESHOLD)

    print(f"  {'Category':<12} {'Count':>6}  {'Dice':>8}")
    print(f"  {'-'*30}")
    for cat, vals in results.items():
        print(f"  {cat:<12} {vals['count']:>6}  {vals['dice']:>8.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  evaluate_negatives  —  FPR on polyp-free images (your clinical novelty)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_negatives(model, neg_loader, cfg):
    """
    Evaluate false positive rate on polyp-free images.
    FPR = fraction of pixels incorrectly predicted as polyp
    when GT is all-zero.

    This is your clinical novelty contribution:
    "First study to evaluate polyp segmentation models on held-out
     negative (polyp-free) colonoscopy images."
    """
    if neg_loader is None:
        print("  [Warning] No negative loader provided.")
        return None

    print(f"\n{'='*60}")
    print(f"  NEGATIVE SAMPLE EVALUATION (Clinical FPR)")
    print(f"{'='*60}")

    model.eval()
    device = cfg.DEVICE
    fpr_values = []

    for imgs, seg_gt, _ in neg_loader:
        imgs   = imgs.to(device)
        seg_gt = seg_gt.cpu()

        seg_logits, _, _, _ = model(imgs)
        prob = torch.sigmoid(seg_logits).cpu()

        for i in range(prob.size(0)):
            fpr = fpr_on_negatives(prob[i], seg_gt[i], cfg.THRESHOLD)
            if fpr is not None:
                fpr_values.append(fpr)

    if not fpr_values:
        print("  [Warning] No true negative samples found.")
        return None

    mean_fpr = float(np.mean(fpr_values))
    std_fpr  = float(np.std(fpr_values))
    print(f"  FPR (negatives): {mean_fpr:.4f} ± {std_fpr:.4f}  "
          f"(n={len(fpr_values)} images)")
    print(f"  Clinical meaning: {(1-mean_fpr)*100:.1f}% of polyp-free frames "
          f"correctly identified as negative")

    return {"mean_fpr": mean_fpr, "std_fpr": std_fpr, "n": len(fpr_values)}


# ══════════════════════════════════════════════════════════════════════════════
#  save_visualizations  —  prediction grid for paper figures
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_visualizations(model, loader, cfg, n_samples=12, save_path=None):
    """
    Save a grid of (image | GT mask | prediction | boundary pred).
    Based on Code 1's show_predictions pattern — extended to 4 columns.
    """
    if save_path is None:
        save_path = os.path.join(cfg.RESULTS_DIR, "predictions.png")

    model.eval()
    device = cfg.DEVICE

    # Collect samples
    samples = []
    for imgs, seg_gt, bnd_gt in loader:
        imgs_d = imgs.to(device)
        seg_logits, bnd_logits, _, _ = model(imgs_d)
        probs = torch.sigmoid(seg_logits).cpu()
        bnds  = torch.sigmoid(bnd_logits).cpu()

        for i in range(imgs.size(0)):
            samples.append({
                "img":  imgs[i].cpu(),
                "gt":   seg_gt[i].cpu(),
                "pred": probs[i].cpu(),
                "bnd":  bnds[i].cpu(),
            })
            if len(samples) >= n_samples:
                break
        if len(samples) >= n_samples:
            break

    n = len(samples)
    fig = plt.figure(figsize=(16, n * 3.5))
    gs  = gridspec.GridSpec(n, 4, figure=fig,
                            wspace=0.05, hspace=0.15)
    col_titles = ["Input Image", "Ground Truth", "Prediction", "Boundary"]
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for row, s in enumerate(samples):
        # Denormalize image
        img_np = s["img"].numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np * std + mean, 0, 1)

        gt_np   = s["gt"].squeeze().numpy()
        pred_np = (s["pred"].squeeze().numpy() > cfg.THRESHOLD).astype(np.float32)
        bnd_np  = s["bnd"].squeeze().numpy()

        dice = compute_all_metrics(s["pred"], s["gt"], cfg.THRESHOLD)["dice"]

        for col, (data, cmap) in enumerate([
            (img_np,  None),
            (gt_np,   "gray"),
            (pred_np, "gray"),
            (bnd_np,  "hot"),
        ]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(data, cmap=cmap)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight="bold", pad=4)
            if col == 2:
                ax.set_title(f"Dice={dice:.3f}", fontsize=9, pad=2)

    plt.suptitle(f"{cfg.PROJECT_NAME} — Predictions", fontsize=14, y=1.01)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [Visualize] Saved {n} samples to: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  print_results_table  —  LaTeX-ready output
# ══════════════════════════════════════════════════════════════════════════════

def print_results_table(results_dict: dict):
    """
    Print a clean results table ready to copy into LaTeX or paper.
    results_dict: {dataset_name: metrics_dict}
    """
    print("\n" + "=" * 80)
    print(f"  {'Dataset':<20} {'Dice':>8} {'IoU':>8} {'Prec':>8} "
          f"{'Recall':>8} {'Spec':>8} {'F1':>8} {'HD95':>8}")
    print("  " + "-" * 78)
    for name, m in results_dict.items():
        print(f"  {name:<20} "
              f"{m.get('dice',0):.4f}  "
              f"{m.get('iou',0):.4f}  "
              f"{m.get('precision',0):.4f}  "
              f"{m.get('recall',0):.4f}  "
              f"{m.get('specificity',0):.4f}  "
              f"{m.get('f1',0):.4f}  "
              f"{m.get('hd95',0):.2f}")
    print("=" * 80 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  run_full_evaluation  —  run everything, save everything
# ══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(model, test_loader, cvc_loader, busi_loader, cfg):
    """
    Master evaluation function — run all evaluation protocols.
    Call this after training to generate everything needed for your paper.
    """
    print(f"\n{'#'*60}")
    print(f"  FULL EVALUATION SUITE — {cfg.PROJECT_NAME}")
    print(f"{'#'*60}")

    all_results = {}

    # 1. Primary test set
    test_metrics, preds, gts = evaluate_on_test(
        model, test_loader, cfg, use_tta=cfg.USE_TTA
    )
    all_results["Kvasir-SEG"] = test_metrics

    # 2. Cross-dataset
    cross_results = evaluate_cross_dataset(model, cvc_loader, busi_loader, cfg)
    all_results.update(cross_results)

    # 3. Size-stratified
    size_results = evaluate_by_size(preds, gts, cfg)

    # 4. Visualizations
    save_visualizations(model, test_loader, cfg, n_samples=12)

    # 5. Print final table
    print_results_table(all_results)

    # 6. Save results to text file
    results_path = os.path.join(cfg.RESULTS_DIR, "final_results.txt")
    with open(results_path, "w") as f:
        f.write(f"{cfg.PROJECT_NAME} — Final Evaluation Results\n")
        f.write("=" * 60 + "\n\n")
        for name, m in all_results.items():
            f.write(f"{name}:\n")
            for k, v in m.items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
        f.write("\nSize-Stratified Results:\n")
        for cat, vals in size_results.items():
            f.write(f"  {cat}: Dice={vals['dice']:.4f}  (n={vals['count']})\n")
    print(f"\n  [Evaluate] Results saved to: {results_path}")

    return all_results, size_results
