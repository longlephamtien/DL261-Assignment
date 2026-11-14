"""Foreground IoU and Dice over (image, expression) pairs. Implemented by #97.

A pair has one foreground class, so the nineteen-way confusion matrix of a semantic task
collapses to three counts. Totals are kept per pair as well as pooled, because the two answer
different questions: pooled counts say how many pixels a model gets right, and the per-pair mean
says how many expressions it resolves.
"""

from __future__ import annotations

import torch
from torch import Tensor


class PairMetrics:
    """Running intersection, union and overlap, accumulated over a split."""

    def __init__(self) -> None:
        self.intersection = 0
        self.union = 0
        self.predicted = 0
        self.target = 0
        self.scores: list[float] = []

    def update(self, targets: Tensor, predictions: Tensor) -> None:
        """Accumulate one batch of (N, H, W) masks holding 0 for background and 1 for the target."""
        if targets.shape != predictions.shape:
            raise ValueError(f"shape mismatch: targets {tuple(targets.shape)}, "
                             f"predictions {tuple(predictions.shape)}")

        hit = (targets == 1) & (predictions == 1)
        either = (targets == 1) | (predictions == 1)
        axes = tuple(range(1, targets.dim()))

        per_pair_hit = hit.sum(dim=axes)
        per_pair_either = either.sum(dim=axes)
        self.intersection += int(per_pair_hit.sum())
        self.union += int(per_pair_either.sum())
        self.predicted += int((predictions == 1).sum())
        self.target += int((targets == 1).sum())

        # A pair whose target and prediction are both empty is perfectly resolved, so it scores 1
        # rather than dividing by zero.
        empty = per_pair_either == 0
        scores = torch.where(empty, torch.ones_like(per_pair_hit, dtype=torch.float64),
                             per_pair_hit.double() / per_pair_either.clamp(min=1))
        self.scores.extend(scores.tolist())

    def mean_iou(self) -> float:
        """Mean of the per-pair IoU, the metric a referring expression is judged by."""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    def pooled_iou(self) -> float:
        """IoU over all pixels of the split, which large targets dominate."""
        return self.intersection / self.union if self.union else 0.0

    def mean_dice(self) -> float:
        """Dice over all pixels of the split."""
        total = self.predicted + self.target
        return 2 * self.intersection / total if total else 0.0

    def precision_at(self, threshold: float = 0.5) -> float:
        """Share of pairs whose IoU clears `threshold`, the convention in this literature."""
        if not self.scores:
            return 0.0
        return sum(1 for score in self.scores if score >= threshold) / len(self.scores)
