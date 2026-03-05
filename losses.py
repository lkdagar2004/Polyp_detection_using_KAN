"""
losses.py  —  KA-ResUNet++
============================
All loss functions in one place.

Sources:
  DiceLoss         → Code 1 (Cell 19), exact copy, smooth=1e-5
  BoundaryLoss     → NEW
  CombinedLoss     → NEW (assembles everything)

Formula:
  total = bce_seg + dice_seg
        + BOUNDARY_WEIGHT  * bce_boundary
        + AUX4_WEIGHT      * bce_aux4
        + AUX3_WEIGHT      * bce_aux3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  DiceLoss  (from Code 1, Cell 19)
# ══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Soft Dice Loss.
    Works on PROBABILITIES (apply sigmoid before calling if using logits).
    smooth=1e-5 prevents division by zero on empty masks.
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.view(-1)
        targets     = targets.view(-1)
        intersection = (predictions * targets).sum()
        union        = predictions.sum() + targets.sum()
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


# ══════════════════════════════════════════════════════════════════════════════
#  BoundaryLoss  (NEW)
# ══════════════════════════════════════════════════════════════════════════════

class BoundaryLoss(nn.Module):
    """
    Binary cross-entropy on boundary maps.
    Boundary GT is precomputed in the dataset class as:
        boundary = dilate(mask) - erode(mask)
    This forces the model to precisely localise polyp edges,
    which is where clinical diagnosis depends.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, boundary_logits: torch.Tensor,
                boundary_gt: torch.Tensor) -> torch.Tensor:
        return self.bce(boundary_logits, boundary_gt)


# ══════════════════════════════════════════════════════════════════════════════
#  CombinedLoss  (NEW — assembles all components)
# ══════════════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    """
    Full loss function for KA-ResUNet++.

    Components:
        L = bce_seg + dice_seg
          + bnd_w  * bce_boundary
          + aux4_w * bce_aux4   (deep supervision at decoder stage 4)
          + aux3_w * bce_aux3   (deep supervision at decoder stage 3)

    Args:
        pos_weight  : weight for positive class in bce_seg (handles imbalance)
        bnd_weight  : weight for boundary loss term
        aux4_weight : weight for deep supervision at stage 4
        aux3_weight : weight for deep supervision at stage 3
        dice_weight : weight for Dice loss term
    """

    def __init__(
        self,
        pos_weight:  float = 2.0,
        bnd_weight:  float = 0.5,
        aux4_weight: float = 0.4,
        aux3_weight: float = 0.2,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        pw = torch.tensor([pos_weight])
        self.bce_seg  = nn.BCEWithLogitsLoss(pos_weight=pw)
        self.bce_bnd  = nn.BCEWithLogitsLoss()
        self.bce_aux  = nn.BCEWithLogitsLoss()
        self.dice     = DiceLoss(smooth=1e-5)
        self.bnd_w    = bnd_weight
        self.aux4_w   = aux4_weight
        self.aux3_w   = aux3_weight
        self.dice_w   = dice_weight

    def forward(
        self,
        seg_logits:  torch.Tensor,   # [B, 1, H, W]   raw seg logits
        bnd_logits:  torch.Tensor,   # [B, 1, H, W]   raw boundary logits
        aux4_logits: torch.Tensor,   # [B, 1, H', W']  deep supervision
        aux3_logits: torch.Tensor,   # [B, 1, H'', W''] deep supervision
        seg_gt:      torch.Tensor,   # [B, 1, H, W]   binary float GT
        bnd_gt:      torch.Tensor,   # [B, 1, H, W]   boundary float GT
    ):
        """
        Returns:
            total_loss : scalar tensor
            loss_dict  : dict with individual loss values for logging
        """
        H, W = seg_gt.shape[2:]

        # Upsample auxiliary outputs to full resolution for loss computation
        aux4_up = F.interpolate(aux4_logits, size=(H, W), mode='bilinear', align_corners=False)
        aux3_up = F.interpolate(aux3_logits, size=(H, W), mode='bilinear', align_corners=False)

        # BCE on segmentation logits (with pos_weight for class imbalance)
        l_bce_seg = self.bce_seg(seg_logits, seg_gt)

        # Soft Dice on probabilities
        l_dice = self.dice(torch.sigmoid(seg_logits), seg_gt) * self.dice_w

        # Boundary supervision
        l_bnd = self.bce_bnd(bnd_logits, bnd_gt) * self.bnd_w

        # Deep supervision
        l_aux4 = self.bce_aux(aux4_up, seg_gt) * self.aux4_w
        l_aux3 = self.bce_aux(aux3_up, seg_gt) * self.aux3_w

        total = l_bce_seg + l_dice + l_bnd + l_aux4 + l_aux3

        loss_dict = {
            "total":   total.item(),
            "bce_seg": l_bce_seg.item(),
            "dice":    l_dice.item(),
            "bnd":     l_bnd.item(),
            "aux4":    l_aux4.item(),
            "aux3":    l_aux3.item(),
        }
        return total, loss_dict


# ── Factory ──────────────────────────────────────────────────────────────────

def build_criterion(cfg) -> CombinedLoss:
    return CombinedLoss(
        pos_weight  = cfg.POS_WEIGHT,
        bnd_weight  = cfg.BOUNDARY_WEIGHT,
        aux4_weight = cfg.AUX4_WEIGHT,
        aux3_weight = cfg.AUX3_WEIGHT,
        dice_weight = cfg.DICE_WEIGHT,
    ).to(cfg.DEVICE)
