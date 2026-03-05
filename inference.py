"""
inference.py  —  KA-ResUNet++
================================
Single-image inference + Test-Time Augmentation (TTA).

Source: Code 4, Cell 50 (predict_tta) — with bug fixed.

BUG in Code 4:
    The transpose pass called model a second time without transposing
    the output back. This produced wrong predictions.

FIX:
    Pass 4: x_t = x.transpose(2,3)
            o   = model(x_t)
            o   = o.transpose(2,3)   ← output transposed back (was missing)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
#  predict_tta  (Code 4 bug-fixed)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_tta(model, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Test-Time Augmentation: average 4 forward passes.
        Pass 1: original
        Pass 2: horizontal flip  → flip output back
        Pass 3: vertical flip    → flip output back
        Pass 4: transpose        → transpose output back  ← BUG FIX from Code 4

    Args:
        model     : KAResUNet in eval mode
        x         : input tensor [B, 3, H, W]
        threshold : sigmoid threshold for binary output

    Returns:
        prob_avg : averaged sigmoid probability [B, 1, H, W]
    """
    model.eval()
    outs = []

    # Pass 1: original
    seg_logits, _, _, _ = model(x)
    outs.append(torch.sigmoid(seg_logits))

    # Pass 2: horizontal flip
    x_hf = torch.flip(x, dims=[3])
    seg_logits_hf, _, _, _ = model(x_hf)
    outs.append(torch.flip(torch.sigmoid(seg_logits_hf), dims=[3]))

    # Pass 3: vertical flip
    x_vf = torch.flip(x, dims=[2])
    seg_logits_vf, _, _, _ = model(x_vf)
    outs.append(torch.flip(torch.sigmoid(seg_logits_vf), dims=[2]))

    # Pass 4: transpose (swap H and W)  ← BUG FIX: must transpose OUTPUT back
    x_t = x.transpose(2, 3).contiguous()
    seg_logits_t, _, _, _ = model(x_t)
    outs.append(torch.sigmoid(seg_logits_t).transpose(2, 3))   # ← fix was here

    prob_avg = torch.stack(outs, dim=0).mean(dim=0)
    return prob_avg


# ══════════════════════════════════════════════════════════════════════════════
#  predict_single  —  inference on one image
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single(
    model,
    img_path: str,
    img_size: int = 256,
    threshold: float = 0.5,
    use_tta: bool = True,
    device: str = "cuda",
) -> np.ndarray:
    """
    Run inference on a single image file.

    Returns:
        pred_mask : binary numpy array [H, W] with values 0/1
    """
    # ── Load & preprocess ──────────────────────────────────────────────────
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    # Resize and normalize
    img_resized = cv2.resize(image, (img_size, img_size)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_norm = (img_resized - mean) / std
    x = torch.from_numpy(img_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    # ── Inference ─────────────────────────────────────────────────────────
    model.eval()
    if use_tta:
        prob = predict_tta(model, x, threshold)
    else:
        seg_logits, _, _, _ = model(x)
        prob = torch.sigmoid(seg_logits)

    # ── Postprocess ───────────────────────────────────────────────────────
    prob_np   = prob.squeeze().cpu().numpy()
    pred_mask = (prob_np > threshold).astype(np.uint8)

    # Resize back to original resolution
    if (orig_h, orig_w) != (img_size, img_size):
        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h),
                               interpolation=cv2.INTER_NEAREST)
    return pred_mask


# ══════════════════════════════════════════════════════════════════════════════
#  overlay_prediction  —  visualization helper
# ══════════════════════════════════════════════════════════════════════════════

def overlay_prediction(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay prediction (and optionally GT) on image.
    - Prediction  → green channel
    - Ground truth → red channel (if provided)
    - Overlap      → yellow

    Args:
        image     : RGB uint8 [H, W, 3]
        pred_mask : binary [H, W]
        gt_mask   : binary [H, W] or None
        alpha     : overlay transparency

    Returns:
        overlay : RGB uint8 [H, W, 3]
    """
    overlay = image.copy().astype(np.float32)
    h, w    = image.shape[:2]

    # Resize masks to image size if needed
    if pred_mask.shape != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if gt_mask is not None and gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    pred_bool = pred_mask.astype(bool)

    if gt_mask is not None:
        gt_bool = gt_mask.astype(bool)
        # True positives → green (pred & gt)
        tp = pred_bool & gt_bool
        # False positives → blue (pred but not gt)
        fp = pred_bool & ~gt_bool
        # False negatives → red (gt but not pred)
        fn = ~pred_bool & gt_bool

        overlay[tp] = overlay[tp] * (1 - alpha) + np.array([0, 255, 0]) * alpha
        overlay[fp] = overlay[fp] * (1 - alpha) + np.array([0, 0, 255]) * alpha
        overlay[fn] = overlay[fn] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    else:
        overlay[pred_bool] = (overlay[pred_bool] * (1 - alpha)
                              + np.array([0, 255, 0]) * alpha)

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  load_model  —  load checkpoint for inference
# ══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, cfg) -> torch.nn.Module:
    """Load model from checkpoint for inference."""
    from models import build_model
    model = build_model(cfg, pretrained=False)
    ckpt  = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  [Inference] Loaded checkpoint: {checkpoint_path}")
    print(f"  [Inference] Best val Dice: {ckpt.get('best_val_dice', 'N/A')}")
    return model
