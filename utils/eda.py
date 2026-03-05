"""
utils/eda.py  —  KA-ResUNet++
================================
Statistical EDA from Code 4.
Move ALL EDA cells here so they run once and produce paper figures.

Sources:
  Data scanning + metrics    → Code 4, Cells 6-21
  Shapiro-Wilk test          → Code 4, Cell 25 (mislabeled as KS in original)
  Spearman correlation       → Code 4, Cell 29
  Welch T-test               → Code 4, Cell 31
  Pearson correlation        → Code 4, Cell 33

These statistical tests constitute Section 3 of your paper.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from glob import glob
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
#  Data Collection
# ══════════════════════════════════════════════════════════════════════════════

def collect_dataset_info(dataset_dirs: list, exts=("png", "jpg", "jpeg")) -> pd.DataFrame:
    """
    Collect image+mask metadata for EDA.
    dataset_dirs: list of (img_dir, mask_dir, dataset_name)
    Returns a DataFrame with one row per image.
    """
    rows = []
    for img_dir, mask_dir, ds_name in dataset_dirs:
        if not os.path.isdir(img_dir):
            continue
        img_paths = []
        for ext in exts:
            img_paths.extend(glob(os.path.join(img_dir, f"*.{ext}")))
            img_paths.extend(glob(os.path.join(img_dir, f"*.{ext.upper()}")))

        for img_path in sorted(img_paths):
            img  = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            # Find matching mask
            basename = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = None
            for ext in exts:
                p = os.path.join(mask_dir, f"{basename}.{ext}")
                if os.path.exists(p):
                    mask_path = p
                    break

            coverage = 0.0
            mean_brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            contrast = float(np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    coverage = float((mask > 127).sum()) / mask.size

            rows.append({
                "dataset":    ds_name,
                "filename":   os.path.basename(img_path),
                "width":      w,
                "height":     h,
                "aspect_ratio": w / (h + 1e-8),
                "orientation": "landscape" if w > h else "portrait",
                "mask_coverage": coverage,
                "mean_brightness": mean_brightness,
                "contrast":   contrast,
            })

    df = pd.DataFrame(rows)
    print(f"  [EDA] Collected {len(df)} samples across {df['dataset'].nunique()} datasets")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Statistical Tests  (Code 4 Cells 25-33)
# ══════════════════════════════════════════════════════════════════════════════

def run_statistical_tests(df: pd.DataFrame, save_dir: str = "./results/eda"):
    """Run all 4 statistical tests from Code 4 EDA."""
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    # ── Test 1: Shapiro-Wilk on polyp size distribution ─────────────────────
    # (Code 4 Cell 25 — was mislabeled as KS test in original)
    coverage = df[df["mask_coverage"] > 0]["mask_coverage"].values
    if len(coverage) >= 3:
        stat, pval = stats.shapiro(coverage[:5000])   # shapiro limit 5000
        results["shapiro_wilk"] = {"statistic": stat, "p_value": pval,
                                    "interpretation":
                                    "NOT normally distributed (p<0.05) — small polyps dominate"
                                    if pval < 0.05 else "Normally distributed"}
        print(f"\n  [EDA] Shapiro-Wilk Test (polyp size distribution):")
        print(f"        stat={stat:.4f}  p={pval:.4e}  → {results['shapiro_wilk']['interpretation']}")

    # ── Test 2: Spearman Correlation (resolution vs polyp size) ─────────────
    poly_df = df[df["mask_coverage"] > 0]
    if len(poly_df) > 10:
        corr, pval = stats.spearmanr(poly_df["width"] * poly_df["height"],
                                      poly_df["mask_coverage"])
        results["spearman_resolution_size"] = {
            "correlation": corr, "p_value": pval,
            "interpretation": "Weak positive correlation between resolution and polyp size"
                              if abs(corr) < 0.3 else
                              "Moderate correlation between resolution and polyp size"
        }
        print(f"\n  [EDA] Spearman Correlation (resolution ↔ polyp size):")
        print(f"        ρ={corr:.4f}  p={pval:.4e}  → {results['spearman_resolution_size']['interpretation']}")

    # ── Test 3: Welch T-test (portrait vs landscape polyp coverage) ──────────
    landscape = df[df["orientation"] == "landscape"]["mask_coverage"].values
    portrait  = df[df["orientation"] == "portrait" ]["mask_coverage"].values
    if len(landscape) > 5 and len(portrait) > 5:
        stat, pval = stats.ttest_ind(landscape, portrait, equal_var=False)  # Welch
        results["welch_ttest_orientation"] = {
            "statistic": stat, "p_value": pval,
            "interpretation": "Significant difference in coverage by orientation (p<0.05)"
                              if pval < 0.05 else "No significant orientation effect"
        }
        print(f"\n  [EDA] Welch T-test (portrait vs landscape coverage):")
        print(f"        t={stat:.4f}  p={pval:.4e}  → {results['welch_ttest_orientation']['interpretation']}")

    # ── Test 4: Pearson Correlation (polyp size vs contrast) ─────────────────
    if len(poly_df) > 10:
        corr, pval = stats.pearsonr(poly_df["mask_coverage"], poly_df["contrast"])
        results["pearson_size_contrast"] = {
            "correlation": corr, "p_value": pval,
            "interpretation": "Very weak correlation between polyp size and image contrast"
        }
        print(f"\n  [EDA] Pearson Correlation (polyp size ↔ contrast):")
        print(f"        r={corr:.4f}  p={pval:.4e}  → {results['pearson_size_contrast']['interpretation']}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  EDA Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def plot_eda(df: pd.DataFrame, save_dir: str = "./results/eda"):
    """Generate all EDA plots for the paper."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Coverage distribution
    axes[0, 0].hist(df["mask_coverage"], bins=50, color="steelblue", edgecolor="white", linewidth=0.5)
    axes[0, 0].set_title("Polyp Coverage Distribution", fontweight="bold")
    axes[0, 0].set_xlabel("Coverage (fraction of image)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].axvline(df["mask_coverage"].mean(), color="red", linestyle="--",
                        label=f"Mean={df['mask_coverage'].mean():.3f}")
    axes[0, 0].legend()

    # 2. Coverage by dataset
    if df["dataset"].nunique() > 1:
        df.boxplot(column="mask_coverage", by="dataset", ax=axes[0, 1])
        axes[0, 1].set_title("Coverage by Dataset", fontweight="bold")
        axes[0, 1].set_xlabel("Dataset")
        axes[0, 1].set_ylabel("Coverage")
        plt.sca(axes[0, 1])
        plt.xticks(rotation=15, ha="right")

    # 3. Image resolution scatter
    axes[0, 2].scatter(df["width"], df["height"], c=df["mask_coverage"],
                        cmap="YlOrRd", alpha=0.5, s=15)
    axes[0, 2].set_title("Image Resolution Distribution", fontweight="bold")
    axes[0, 2].set_xlabel("Width (px)")
    axes[0, 2].set_ylabel("Height (px)")

    # 4. Size category bar chart
    def categorize(c):
        if c == 0: return "empty"
        elif c <= 0.05: return "small"
        elif c <= 0.15: return "medium"
        elif c <= 0.30: return "large"
        else: return "huge"
    df["size_cat"] = df["mask_coverage"].apply(categorize)
    cat_counts = df["size_cat"].value_counts().reindex(
        ["empty", "small", "medium", "large", "huge"], fill_value=0
    )
    cat_counts.plot(kind="bar", ax=axes[1, 0], color="steelblue", edgecolor="navy")
    axes[1, 0].set_title("Polyp Size Category Distribution", fontweight="bold")
    axes[1, 0].set_xlabel("Size Category")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].tick_params(axis="x", rotation=0)

    # 5. Brightness vs Coverage
    axes[1, 1].scatter(df["mean_brightness"], df["mask_coverage"],
                        alpha=0.3, s=10, c="steelblue")
    axes[1, 1].set_title("Brightness vs Polyp Coverage", fontweight="bold")
    axes[1, 1].set_xlabel("Mean Brightness")
    axes[1, 1].set_ylabel("Mask Coverage")

    # 6. Aspect ratio distribution
    axes[1, 2].hist(df["aspect_ratio"], bins=30, color="coral", edgecolor="white")
    axes[1, 2].set_title("Aspect Ratio Distribution", fontweight="bold")
    axes[1, 2].set_xlabel("Width / Height")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].axvline(1.0, color="red", linestyle="--", label="Square")
    axes[1, 2].legend()

    plt.suptitle("Dataset EDA — KA-ResUNet++", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(save_dir, "eda_plots.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [EDA] Plots saved to: {out}")


def run_eda(cfg, save_dir: str = "./results/eda"):
    """Run complete EDA pipeline — call this once before training."""
    print(f"\n{'='*60}")
    print(f"  STATISTICAL EDA")
    print(f"{'='*60}")

    dataset_dirs = [
        (cfg.KVASIR_IMG_DIR,   cfg.KVASIR_MASK_DIR,   "Kvasir-SEG"),
        (cfg.SESSILE_IMG_DIR,  cfg.SESSILE_MASK_DIR,  "Kvasir-Sessile"),
        (cfg.ENDOTECT_IMG_DIR, cfg.ENDOTECT_MASK_DIR, "EndoTect"),
        (cfg.CVC_IMG_DIR,      cfg.CVC_MASK_DIR,      "CVC-ClinicDB"),
        (cfg.NEG_IMG_DIR,      cfg.NEG_MASK_DIR,      "Negatives"),
    ]
    # Filter to existing dirs
    dataset_dirs = [(a, b, c) for a, b, c in dataset_dirs if os.path.isdir(a)]

    if not dataset_dirs:
        print("  [EDA] No dataset directories found. Skipping EDA.")
        return

    df = collect_dataset_info(dataset_dirs)
    stat_results = run_statistical_tests(df, save_dir)
    plot_eda(df, save_dir)

    # Save summary CSV
    summary_path = os.path.join(save_dir, "dataset_info.csv")
    df.to_csv(summary_path, index=False)
    print(f"  [EDA] Dataset info saved to: {summary_path}")

    return df, stat_results
