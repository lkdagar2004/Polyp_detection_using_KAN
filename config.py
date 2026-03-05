"""
config.py  —  KA-ResUNet++
Single source of truth for ALL hyperparameters.
No magic numbers anywhere else in the codebase.
"""
import os
import torch


class Config:
    # ── GENERAL ──────────────────────────────────────────────────────────────
    PROJECT_NAME        = "KA-ResUNet++"
    SEED                = 42
    DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS         = 4
    PIN_MEMORY          = True

    # ── IMAGE ─────────────────────────────────────────────────────────────────
    IMG_SIZE            = 256
    IN_CHANNELS         = 3

    # ── MODEL ─────────────────────────────────────────────────────────────────
    NUM_CLASSES         = 1
    EMBED_DIMS          = [128, 160, 256]   # KAN bottleneck dims [dec4, dec3, bottleneck]
    DROP_RATE           = 0.1
    DROP_PATH_RATE      = 0.1
    KAN_GRID_SIZE       = 5
    KAN_SPLINE_ORDER    = 3
    KAN_NUM_GRIDS       = 4
    KAN_GRID_MIN        = -2.0
    KAN_GRID_MAX        = 2.0

    # ── TRAINING ──────────────────────────────────────────────────────────────
    BATCH_SIZE          = 8
    VAL_BATCH_SIZE      = 4
    NUM_EPOCHS          = 100
    LR                  = 1e-4
    KAN_LR              = 5e-5       # KAN fc layers get lower lr
    WEIGHT_DECAY        = 1e-4
    GRAD_CLIP           = 1.0
    EARLY_STOP_PATIENCE = 15
    MIXED_PRECISION     = True

    # Scheduler
    SCHEDULER_T0        = 10
    SCHEDULER_T_MULT    = 2
    SCHEDULER_ETA_MIN   = 1e-6

    # ── LOSS WEIGHTS ──────────────────────────────────────────────────────────
    POS_WEIGHT          = 2.0
    BOUNDARY_WEIGHT     = 0.5
    AUX4_WEIGHT         = 0.4
    AUX3_WEIGHT         = 0.2
    DICE_WEIGHT         = 1.0

    # ── METRICS ───────────────────────────────────────────────────────────────
    THRESHOLD           = 0.5
    SMOOTH              = 1e-5

    # ── TTA ───────────────────────────────────────────────────────────────────
    USE_TTA             = True

    # ── DATASET PATHS ─────────────────────────────────────────────────────────
    KVASIR_IMG_DIR      = "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/images"
    KVASIR_MASK_DIR     = "/kaggle/input/kvasirseg/Kvasir-SEG/Kvasir-SEG/masks"

    CVC_IMG_DIR         = "/kaggle/input/cvcclinicdb/PNG/Original"
    CVC_MASK_DIR        = "/kaggle/input/cvcclinicdb/PNG/Ground Truth"

    SESSILE_IMG_DIR     = "/kaggle/input/kvasirsessile/sessile-main-Kvasir-SEG/images"
    SESSILE_MASK_DIR    = "/kaggle/input/kvasirsessile/sessile-main-Kvasir-SEG/masks"

    ENDOTECT_IMG_DIR    = "/kaggle/input/endotect-dataset/EndoTect/Training dataset (Kvasir-SEG)/images"
    ENDOTECT_MASK_DIR   = "/kaggle/input/endotect-dataset/EndoTect/Training dataset (Kvasir-SEG)/masks"

    NEG_IMG_DIR         = "/kaggle/working/neg_samples/content/neg_samples/images"
    NEG_MASK_DIR        = "/kaggle/working/neg_samples/content/neg_samples/masks"

    BUSI_IMG_DIR        = "/kaggle/input/test-busi-paper/Original"
    BUSI_MASK_DIR       = "/kaggle/input/test-busi-paper/Ground Truth"

    # ── SPLITS ────────────────────────────────────────────────────────────────
    TRAIN_RATIO         = 0.80
    VAL_RATIO           = 0.10
    TEST_RATIO          = 0.10

    # ── PATHS ─────────────────────────────────────────────────────────────────
    CHECKPOINT_DIR      = "./checkpoints"
    RESULTS_DIR         = "./results"
    BEST_MODEL          = "./checkpoints/best_model.pth"
    LOG_CSV             = "./results/training_log.csv"

    BOUNDARY_KERNEL_SIZE = 5

    @classmethod
    def make_dirs(cls):
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.RESULTS_DIR,    exist_ok=True)

    @classmethod
    def print_summary(cls):
        print("=" * 60)
        print(f"  {cls.PROJECT_NAME}  —  Configuration")
        print("=" * 60)
        print(f"  Device        : {cls.DEVICE}")
        print(f"  Image Size    : {cls.IMG_SIZE}×{cls.IMG_SIZE}")
        print(f"  Batch Size    : {cls.BATCH_SIZE}")
        print(f"  Epochs        : {cls.NUM_EPOCHS}")
        print(f"  LR (CNN)      : {cls.LR}")
        print(f"  LR (KAN fc)   : {cls.KAN_LR}")
        print(f"  Mixed Prec.   : {cls.MIXED_PRECISION}")
        print(f"  TTA           : {cls.USE_TTA}")
        print("=" * 60)


cfg = Config()
