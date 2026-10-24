"""Segmentation metrics: mIoU and Dice, accumulated over a confusion matrix."""

from __future__ import annotations

import torch
from torch import Tensor

from .interfaces import IGNORE_INDEX, NUM_CLASSES


class ConfusionMatrix:
    """Accumulates predictions over a split; every metric is derived from it afterwards."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, targets: Tensor, predictions: Tensor) -> None:
        """Add one batch, dropping pixels labelled IGNORE_INDEX.

        Rows are the true class and columns the prediction. Cells are counted with a single
        `bincount` over `target * num_classes + prediction`, which keeps the cost proportional
        to the pixels in the batch rather than to the number of classes squared.
        """
        if targets.shape != predictions.shape:
            raise ValueError(f"targets {tuple(targets.shape)} and predictions {tuple(predictions.shape)} differ")

        targets = targets.flatten().to(torch.int64)
        predictions = predictions.flatten().to(torch.int64)

        keep = targets != IGNORE_INDEX
        targets, predictions = targets[keep], predictions[keep]
        if targets.numel() == 0:
            return

        if (targets >= self.num_classes).any() or (targets < 0).any():
            raise ValueError(f"targets hold a label outside [0, {self.num_classes}) and not IGNORE_INDEX")
        if (predictions >= self.num_classes).any() or (predictions < 0).any():
            raise ValueError(f"predictions hold a class outside [0, {self.num_classes})")

        counts = torch.bincount(
            targets * self.num_classes + predictions, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        self.matrix += counts.to(self.matrix.device)

    def _parts(self) -> tuple[Tensor, Tensor, Tensor]:
        """True positives, and the per-class totals of the true and predicted labels."""
        matrix = self.matrix.to(torch.float64)
        return matrix.diag(), matrix.sum(dim=1), matrix.sum(dim=0)

    def iou(self) -> Tensor:
        """Per-class intersection over union, NaN where a class is absent from the split."""
        intersection, actual, predicted = self._parts()
        union = actual + predicted - intersection
        return torch.where(union > 0, intersection / union, torch.nan)

    def dice(self) -> Tensor:
        """Per-class Dice coefficient, NaN where a class is absent from the split."""
        intersection, actual, predicted = self._parts()
        total = actual + predicted
        return torch.where(total > 0, 2 * intersection / total, torch.nan)

    def pixel_accuracy(self) -> float:
        """Share of evaluated pixels predicted correctly, ignored pixels excluded."""
        intersection, actual, _ = self._parts()
        evaluated = actual.sum()
        return float(intersection.sum() / evaluated) if evaluated > 0 else 0.0

    def mean_iou(self) -> float:
        """mIoU over the classes present in the split."""
        return float(torch.nanmean(self.iou()))

    def mean_dice(self) -> float:
        """Mean Dice over the classes present in the split."""
        return float(torch.nanmean(self.dice()))
