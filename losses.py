"""
Loss functions for binary brain tumor segmentation (RQ2 comparison).

All losses take:
  - logits: float32 tensor of shape `(B, 1, H, W)` (raw U-Net output before sigmoid)
  - targets: float32 tensor of shape `(B, 1, H, W)` with binary values `{0.0, 1.0}`
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BinarySoftDiceLoss(nn.Module):
    """
    Batch-level differentiable Soft Dice Loss for imbalanced binary segmentation.

    Summing over `(B, H, W)` stabilizes training when individual 2D slices contain
    no tumor pixels, and `smooth` prevents division by zero on all-empty batches.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets, dim=dims)
        cardinality = torch.sum(probs + targets, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice_score).mean()


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy + Soft Dice Loss."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = BinarySoftDiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


def get_loss_fn(name: str) -> nn.Module:
    """Factory returning the requested binary segmentation loss function."""
    key = name.strip().lower()
    if key == "bce":
        return nn.BCEWithLogitsLoss()
    if key == "dice":
        return BinarySoftDiceLoss()
    if key == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"Unknown loss '{name}'. Choose from: 'bce', 'dice', 'bce_dice'.")
