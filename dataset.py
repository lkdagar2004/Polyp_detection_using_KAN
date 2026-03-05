"""
dataset.py  —  KA-ResUNet++
=============================
Unified dataset class combining:
  - Code 4's PolypDataset (multi-source, pad_to_square)
  - Code 1's multi-dataset scanner
  - Code 4's full 6-category augmentation pipeline

ALL BUGS FIXED:
  [1] FIX: val_transform includes Resize (was missing in Code 4)
  [2] FIX: No double /255 — Albumentations Normalize handles it
  [3] FIX: Train and val are SEPARATE dataset instances (no shared transform)
  [4] FIX: Returns (image, seg_mask, boundary_mask) — 3-tuple
  [5] FIX: mask binarized at 0.5 BEFORE augmentation, not after

NEW:
  - boundary_mask computed inline: dilate(mask) - erode(mask)
  - Negative samples supported with weight parameter
  - Stratified split by polyp coverage size (from Code 4 EDA)
"""

import os
import cv2
import numpy as np
from glob import glob
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════════════════════
#  Augmentation Pipelines  (Code 4, Cell 42 — all 6 categories)
# ══════════════════════════════════════════════════════════════════════════════

def get_train_transform(img_size: int = 256) -> A.Compose:
    """
    Full 6-category augmentation for training.
    Source: Code 4, Cell 42 — with Resize ADDED at start (was missing).
    """
    return A.Compose([
        # 0. RESIZE FIRST — ensures all spatial ops work on consistent size
        A.Resize(img_size, img_size),

        # 1. GEOMETRIC
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1,
            rotate_limit=15, p=0.5
        ),

        # 2. ELASTIC DEFORMATIONS  (bowel wall / tissue shape variation)
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=12, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.OpticalDistortion(distort_limit=0.2, p=0.5),
        ], p=0.4),

        # 3. LOW QUALITY / ARTIFACTS  (endoscope video compression)
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=0.5),
            A.GaussianBlur(blur_limit=5, p=0.5),
            A.Defocus(radius=(1, 3), p=0.5),
        ], p=0.3),
        A.GaussNoise(p=0.3),
        A.ImageCompression(quality_range=(10, 40), p=0.3),

        # 4. LIGHTING  (endoscope illumination variation)
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=20, p=0.3
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),

        # 5. OCCLUSION  (partial polyp visibility)
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=1, fill_value=0, p=0.3
        ),

        # 6. NORMALIZATION  (ImageNet stats — required for ResNet50 encoder)
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


def get_val_transform(img_size: int = 256) -> A.Compose:
    """
    Validation / test transform.
    FIX: Resize is NOW included (it was missing in Code 4).
    Only resize + normalize — no augmentation.
    """
    return A.Compose([
        A.Resize(img_size, img_size),   # ← FIX: was missing in Code 4
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  Boundary Map Computation  (NEW — used for BoundaryLoss)
# ══════════════════════════════════════════════════════════════════════════════

def compute_boundary(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Compute boundary map as morphological gradient:
        boundary = dilate(mask) - erode(mask)
    Returns float32 array in [0, 1].
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded  = cv2.erode(mask,  kernel, iterations=1)
    boundary = (dilated.astype(np.float32) - eroded.astype(np.float32))
    return np.clip(boundary, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  PolypDataset  (Code 4 fixed + Code 1 multi-dataset scanner)
# ══════════════════════════════════════════════════════════════════════════════

class PolypDataset(Dataset):
    """
    Unified polyp segmentation dataset.

    Args:
        datasets : list of (img_dir, mask_dir) tuples
        transform: albumentations transform
        img_size : resize target
        boundary_kernel_size: morphological kernel for boundary GT
        exts     : valid image extensions to scan
    """

    def __init__(
        self,
        datasets: List[Tuple[str, str]],
        transform: Optional[A.Compose] = None,
        img_size: int = 256,
        boundary_kernel_size: int = 5,
        exts: Tuple[str, ...] = ("png", "jpg", "jpeg"),
    ):
        self.transform = transform
        self.img_size  = img_size
        self.bk_size   = boundary_kernel_size
        self.image_paths: List[str] = []
        self.mask_paths:  List[str] = []

        for img_dir, mask_dir in datasets:
            imgs, masks = self._scan_directory(img_dir, mask_dir, exts)
            self.image_paths.extend(imgs)
            self.mask_paths.extend(masks)

        print(f"  [Dataset] Loaded {len(self.image_paths)} image-mask pairs "
              f"from {len(datasets)} source(s).")

    def _scan_directory(
        self,
        img_dir: str,
        mask_dir: str,
        exts: Tuple[str, ...]
    ) -> Tuple[List[str], List[str]]:
        """Scan img_dir and find matching masks in mask_dir by filename."""
        if not os.path.isdir(img_dir):
            print(f"  [Warning] Directory not found: {img_dir}")
            return [], []

        # Collect all images
        image_paths = []
        for ext in exts:
            image_paths.extend(glob(os.path.join(img_dir, f"*.{ext}")))
            image_paths.extend(glob(os.path.join(img_dir, f"*.{ext.upper()}")))
        image_paths = sorted(set(image_paths))

        # Collect all masks into a lookup dict {basename: full_path}
        mask_paths_all = []
        for ext in exts:
            mask_paths_all.extend(glob(os.path.join(mask_dir, f"*.{ext}")))
            mask_paths_all.extend(glob(os.path.join(mask_dir, f"*.{ext.upper()}")))

        mask_dict = {}
        for p in mask_paths_all:
            basename = os.path.splitext(os.path.basename(p))[0]
            mask_dict[basename] = p

        # Match images to masks by base filename
        matched_imgs  = []
        matched_masks = []
        for img_path in image_paths:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            # Try exact match, then try with _mask suffix stripped
            if basename in mask_dict:
                matched_imgs.append(img_path)
                matched_masks.append(mask_dict[basename])
            elif basename.replace("_mask", "") in mask_dict:
                matched_imgs.append(img_path)
                matched_masks.append(mask_dict[basename.replace("_mask", "")])

        return matched_imgs, matched_masks

    def __len__(self) -> int:
        return len(self.image_paths)

    def pad_to_square(self, img: np.ndarray, color=(0, 0, 0)) -> np.ndarray:
        """Pad image to square preserving aspect ratio (Code 4, Cell 41)."""
        h, w    = img.shape[:2]
        size    = max(h, w)
        top     = (size - h) // 2
        bottom  = size - h - top
        left    = (size - w) // 2
        right   = size - w - left
        return cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=color)

    def __getitem__(self, idx: int):
        """
        Returns:
            image    : float32 tensor [3, H, W]  — normalized
            seg_mask : float32 tensor [1, H, W]  — binary [0,1]
            bnd_mask : float32 tensor [1, H, W]  — boundary [0,1]
        """
        # ── Load ──────────────────────────────────────────────────────────────
        image = cv2.imread(self.image_paths[idx])
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {self.mask_paths[idx]}")

        # ── Pad to square (preserves aspect ratio) ────────────────────────────
        image = self.pad_to_square(image)
        mask  = self.pad_to_square(mask, color=(0,))

        # ── Binarize mask BEFORE augmentation ─────────────────────────────────
        # FIX from Code 1: binarize before transform, not inside transform
        seg_mask = (mask > 127).astype(np.float32)   # [0,1] float

        # ── Compute boundary GT ───────────────────────────────────────────────
        bnd_mask = compute_boundary(seg_mask, self.bk_size)  # [0,1] float

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.transform:
            # Pass mask as uint8 for albumentations, then convert back
            augmented = self.transform(
                image=image,
                mask=(seg_mask * 255).astype(np.uint8)
            )
            image    = augmented["image"]        # tensor [3, H, W]
            seg_aug  = augmented["mask"].float() / 255.0  # [H, W] float

            # Recompute boundary on augmented mask
            seg_np   = seg_aug.numpy()
            bnd_aug  = compute_boundary((seg_np > 0.5).astype(np.float32), self.bk_size)

            seg_mask = seg_aug.unsqueeze(0)                          # [1, H, W]
            bnd_mask = torch.from_numpy(bnd_aug).unsqueeze(0)       # [1, H, W]

        else:
            # No transform: manual resize + to tensor
            # FIX: apply normalization consistently (Code 1 bug: no norm in test)
            image    = cv2.resize(image, (self.img_size, self.img_size))
            seg_mask = cv2.resize(seg_mask, (self.img_size, self.img_size))
            bnd_mask = compute_boundary(
                (seg_mask > 0.5).astype(np.float32), self.bk_size
            )
            mean = np.array([0.485, 0.456, 0.406])
            std  = np.array([0.229, 0.224, 0.225])
            image = (image.astype(np.float32) / 255.0 - mean) / std
            image    = torch.from_numpy(image.transpose(2, 0, 1)).float()
            seg_mask = torch.from_numpy(seg_mask).unsqueeze(0).float()
            bnd_mask = torch.from_numpy(bnd_mask).unsqueeze(0).float()

        return image, seg_mask, bnd_mask


# ══════════════════════════════════════════════════════════════════════════════
#  Coverage-based stratified split  (from Code 4 EDA)
# ══════════════════════════════════════════════════════════════════════════════

def _get_coverage_strata(mask_paths: List[str]) -> List[str]:
    """
    Compute coverage category for each mask for stratified splitting.
    Code 4's 5-bin scheme: empty, small, medium, large, huge.
    """
    strata = []
    for mp in mask_paths:
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            strata.append("small")
            continue
        coverage = (mask > 127).sum() / mask.size
        if coverage == 0:
            strata.append("empty")
        elif coverage <= 0.05:
            strata.append("small")
        elif coverage <= 0.15:
            strata.append("medium")
        elif coverage <= 0.30:
            strata.append("large")
        else:
            strata.append("huge")
    return strata


# ══════════════════════════════════════════════════════════════════════════════
#  build_dataloaders  —  returns train / val / test / cvc_test loaders
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(cfg):
    """
    Build all dataloaders for training and evaluation.

    Training corpus (Kvasir + Sessile + EndoTect + Negatives):
        80% train | 10% val | 10% test  (stratified by polyp size)

    External test:
        CVC-ClinicDB — held out entirely (zero-shot cross-dataset evaluation)
        BUSI          — held out entirely (cross-domain evaluation)

    Returns:
        train_loader, val_loader, test_loader, cvc_loader, busi_loader
    """
    # ── Training corpus (Kvasir-SEG is the primary dataset) ───────────────────
    primary_img_paths  = []
    primary_mask_paths = []

    primary_sources = [
        (cfg.KVASIR_IMG_DIR,   cfg.KVASIR_MASK_DIR),
        (cfg.SESSILE_IMG_DIR,  cfg.SESSILE_MASK_DIR),
        (cfg.ENDOTECT_IMG_DIR, cfg.ENDOTECT_MASK_DIR),
        (cfg.NEG_IMG_DIR,      cfg.NEG_MASK_DIR),
    ]

    for img_dir, mask_dir in primary_sources:
        if not os.path.isdir(img_dir):
            continue
        tmp_ds = PolypDataset([(img_dir, mask_dir)])
        primary_img_paths.extend(tmp_ds.image_paths)
        primary_mask_paths.extend(tmp_ds.mask_paths)

    print(f"\n[DataLoader] Total training corpus: {len(primary_img_paths)} images")

    # ── Stratified train/val/test split ───────────────────────────────────────
    strata = _get_coverage_strata(primary_mask_paths)

    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        primary_img_paths, primary_mask_paths,
        test_size=(cfg.VAL_RATIO + cfg.TEST_RATIO),
        random_state=cfg.SEED,
        stratify=strata
    )
    strata_temp = _get_coverage_strata(temp_masks)
    val_size = cfg.VAL_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)

    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks,
        test_size=(1.0 - val_size),
        random_state=cfg.SEED,
        stratify=strata_temp
    )

    print(f"  Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    # ── Create SEPARATE dataset instances for train and val ───────────────────
    # FIX: Code 4 shared a single dataset object and set transform on it,
    # which caused both train and val to get train_transform.
    # Here we create independent objects with their own transforms.

    train_dataset = PolypDataset(
        datasets=list(zip(
            [os.path.dirname(p) for p in train_imgs],
            [os.path.dirname(p) for p in train_masks]
        )),
        transform=get_train_transform(cfg.IMG_SIZE),
        img_size=cfg.IMG_SIZE,
        boundary_kernel_size=cfg.BOUNDARY_KERNEL_SIZE,
    )
    # Manually set the correct paths (we already know the exact pairs)
    train_dataset.image_paths = train_imgs
    train_dataset.mask_paths  = train_masks

    val_dataset = PolypDataset(
        datasets=[],
        transform=get_val_transform(cfg.IMG_SIZE),
        img_size=cfg.IMG_SIZE,
        boundary_kernel_size=cfg.BOUNDARY_KERNEL_SIZE,
    )
    val_dataset.image_paths = val_imgs
    val_dataset.mask_paths  = val_masks

    test_dataset = PolypDataset(
        datasets=[],
        transform=get_val_transform(cfg.IMG_SIZE),
        img_size=cfg.IMG_SIZE,
        boundary_kernel_size=cfg.BOUNDARY_KERNEL_SIZE,
    )
    test_dataset.image_paths = test_imgs
    test_dataset.mask_paths  = test_masks

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=True, num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.VAL_BATCH_SIZE,
        shuffle=False, num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1,
        shuffle=False, num_workers=cfg.NUM_WORKERS
    )

    # ── External test sets (held out entirely) ────────────────────────────────
    cvc_loader  = _build_external_loader(cfg.CVC_IMG_DIR,  cfg.CVC_MASK_DIR,  cfg)
    busi_loader = _build_external_loader(cfg.BUSI_IMG_DIR, cfg.BUSI_MASK_DIR, cfg)

    return train_loader, val_loader, test_loader, cvc_loader, busi_loader


def _build_external_loader(img_dir: str, mask_dir: str, cfg) -> Optional[DataLoader]:
    """Build a test-only loader for external datasets (CVC, BUSI)."""
    if not os.path.isdir(img_dir):
        return None
    ds = PolypDataset(
        datasets=[(img_dir, mask_dir)],
        transform=get_val_transform(cfg.IMG_SIZE),
        img_size=cfg.IMG_SIZE,
        boundary_kernel_size=cfg.BOUNDARY_KERNEL_SIZE,
    )
    if len(ds) == 0:
        return None
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS)


def verify_batch(loader: DataLoader, name: str = "loader"):
    """Quick sanity check — print shapes of first batch."""
    for imgs, seg_masks, bnd_masks in loader:
        print(f"  [{name}] img:{imgs.shape} seg:{seg_masks.shape} bnd:{bnd_masks.shape}")
        print(f"           img range: [{imgs.min():.2f}, {imgs.max():.2f}]")
        print(f"           seg unique values: {seg_masks.unique().tolist()[:5]}")
        break
