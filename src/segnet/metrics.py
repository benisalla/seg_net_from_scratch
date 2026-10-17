"""Loss (cross-entropy + optional soft dice) and metrics (pixel accuracy, IoU)."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .data import IGNORE_INDEX


def soft_dice_loss(logits: Tensor, target: Tensor, ignore_index: int = IGNORE_INDEX) -> Tensor:
    """1 - mean multi-class soft dice over the classes present in the batch.

    logits: (B, C, H, W); target: (B, H, W) class ids. Ignored pixels are masked out.
    """
    num_classes = logits.shape[1]
    valid = (target != ignore_index).unsqueeze(1)
    probs = logits.float().softmax(dim=1) * valid
    onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2) * valid
    dims = (0, 2, 3)
    intersection = (probs * onehot).sum(dims)
    denominator = probs.sum(dims) + onehot.sum(dims)
    present = onehot.sum(dims) > 0
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    return 1 - dice[present].mean() if present.any() else logits.sum() * 0


class SegmentationLoss(nn.Module):
    """cross_entropy + dice_weight * soft_dice, both ignoring IGNORE_INDEX pixels."""

    def __init__(self, dice_weight: float = 0.0, ignore_index: int = IGNORE_INDEX) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        loss = F.cross_entropy(logits.float(), target, ignore_index=self.ignore_index)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * soft_dice_loss(logits, target, self.ignore_index)
        return loss


class ConfusionMatrix:
    """Accumulates a (C x C) confusion matrix (rows = truth, cols = prediction)."""

    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, pred: Tensor, target: Tensor) -> None:
        valid = target != self.ignore_index
        idx = target[valid] * self.num_classes + pred[valid]
        counts = torch.bincount(idx.cpu(), minlength=self.num_classes**2)
        self.matrix += counts.view(self.num_classes, self.num_classes)

    def iou(self) -> Tensor:
        """IoU per class; NaN for classes that never appear (in truth or prediction)."""
        m = self.matrix.double()
        tp = m.diag()
        union = m.sum(0) + m.sum(1) - tp
        return torch.where(union > 0, tp / union.clamp(min=1), torch.nan)

    def compute(self) -> dict[str, float]:
        m = self.matrix.double()
        iou = self.iou()
        return {
            "pixel_acc": (m.diag().sum() / m.sum().clamp(min=1)).item(),
            "miou": iou.nanmean().item() if not iou.isnan().all() else 0.0,
        }
