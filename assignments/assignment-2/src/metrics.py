"""Segmentation metrics: mIoU and Dice, accumulated over a confusion matrix.

A running confusion matrix is the only state a whole split needs, so memory does not grow
with the number of images. Implemented by #63.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .interfaces import NUM_CLASSES


class ConfusionMatrix:
    """Accumulates predictions over a split; every metric is derived from it afterwards."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(self, targets: Tensor, predictions: Tensor) -> None:
        """Add one batch, dropping pixels labelled IGNORE_INDEX. Implemented by #63."""
        raise NotImplementedError("#63 implements the metric accumulation")

    def iou(self) -> Tensor:
        """Per-class intersection over union. Implemented by #63."""
        raise NotImplementedError("#63 implements the metric accumulation")

    def dice(self) -> Tensor:
        """Per-class Dice coefficient. Implemented by #63."""
        raise NotImplementedError("#63 implements the metric accumulation")

    def pixel_accuracy(self) -> float:
        """Share of evaluated pixels predicted correctly. Implemented by #63."""
        raise NotImplementedError("#63 implements the metric accumulation")
