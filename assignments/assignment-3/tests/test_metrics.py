"""The metric contract, checked against cases computed by hand."""

from __future__ import annotations

import pytest
import torch

from src.metrics import PairMetrics


def masks(*rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def test_a_perfect_prediction_scores_one() -> None:
    target = masks([[1, 1], [0, 0]])
    metrics = PairMetrics()
    metrics.update(target, target.clone())
    assert metrics.mean_iou() == 1.0
    assert metrics.mean_dice() == 1.0


def test_a_disjoint_prediction_scores_zero() -> None:
    metrics = PairMetrics()
    metrics.update(masks([[1, 1], [0, 0]]), masks([[0, 0], [1, 1]]))
    assert metrics.mean_iou() == 0.0
    assert metrics.mean_dice() == 0.0


def test_half_overlap_matches_the_hand_computed_value() -> None:
    """Two target pixels, two predicted, one shared: IoU 1/3, Dice 2/4."""
    metrics = PairMetrics()
    metrics.update(masks([[1, 1], [0, 0]]), masks([[1, 0], [1, 0]]))
    assert metrics.mean_iou() == pytest.approx(1 / 3)
    assert metrics.mean_dice() == pytest.approx(0.5)


def test_an_empty_pair_counts_as_resolved() -> None:
    """Nothing to find and nothing predicted is a correct answer, not a division by zero."""
    metrics = PairMetrics()
    metrics.update(masks([[0, 0], [0, 0]]), masks([[0, 0], [0, 0]]))
    assert metrics.mean_iou() == 1.0


def test_the_per_pair_mean_differs_from_the_pooled_value() -> None:
    """A large easy target must not drown a small failed one, which is why both are kept."""
    targets = torch.zeros(2, 4, 4, dtype=torch.long)
    predictions = torch.zeros(2, 4, 4, dtype=torch.long)
    targets[0] = 1                      # a large target, predicted perfectly
    predictions[0] = 1
    targets[1, 0, 0] = 1                # a single-pixel target, missed entirely
    predictions[1, 3, 3] = 1

    metrics = PairMetrics()
    metrics.update(targets, predictions)
    # Per pair: 1.0 and 0.0, so the mean is 0.5. Pooled: 16 shared pixels over a union of 18.
    assert metrics.mean_iou() == pytest.approx(0.5)
    assert metrics.pooled_iou() == pytest.approx(16 / 18)


def test_precision_at_counts_pairs_over_the_threshold() -> None:
    metrics = PairMetrics()
    metrics.update(masks([[1, 1], [0, 0]]), masks([[1, 1], [0, 0]]))   # IoU 1.0
    metrics.update(masks([[1, 1], [0, 0]]), masks([[1, 0], [1, 0]]))   # IoU 1/3
    assert metrics.precision_at(0.5) == pytest.approx(0.5)
    assert metrics.precision_at(0.3) == pytest.approx(1.0)


def test_mismatched_shapes_are_refused() -> None:
    metrics = PairMetrics()
    with pytest.raises(ValueError, match="shape mismatch"):
        metrics.update(torch.zeros(1, 4, 4, dtype=torch.long), torch.zeros(1, 8, 8, dtype=torch.long))
