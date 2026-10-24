"""Evaluation contract, checked against a synthetic loader rather than a trained run."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import predict
from src.interfaces import IGNORE_INDEX, NUM_CLASSES
from src.metrics import ConfusionMatrix


class Constant(nn.Module):
    """Predicts one fixed class everywhere, so the expected matrix is known in advance."""

    def __init__(self, class_index: int) -> None:
        super().__init__()
        self.class_index = class_index
        self.weight = nn.Parameter(torch.zeros(1))  # keeps the module non-empty

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = torch.full((images.shape[0], NUM_CLASSES, *images.shape[2:]), -1.0)
        logits[:, self.class_index] = 1.0
        return logits


def loader(images: torch.Tensor, targets: torch.Tensor, batch_size: int = 2) -> DataLoader:
    return DataLoader(TensorDataset(images, targets), batch_size=batch_size)


def test_predict_accumulates_over_the_whole_split() -> None:
    """Every pixel of every batch must reach the matrix, not just the last batch."""
    images = torch.randn(6, 3, 8, 8)
    targets = torch.randint(0, NUM_CLASSES, (6, 8, 8))

    matrix, _ = predict(Constant(0), loader(images, targets), torch.device("cpu"))

    assert matrix.matrix.sum().item() == 6 * 8 * 8
    assert matrix.matrix[:, 1:].sum().item() == 0, "a constant model predicts one column only"


def test_predict_drops_ignored_pixels() -> None:
    images = torch.randn(4, 3, 8, 8)
    targets = torch.full((4, 8, 8), IGNORE_INDEX)
    targets[:, :4] = 0

    matrix, _ = predict(Constant(0), loader(images, targets), torch.device("cpu"))

    assert matrix.matrix.sum().item() == 4 * 4 * 8, "only the labelled half is evaluated"
    assert matrix.pixel_accuracy() == 1.0


def test_predict_matches_a_matrix_built_by_hand() -> None:
    """The plumbing must agree with scoring the same predictions directly."""
    images = torch.randn(4, 3, 8, 8)
    targets = torch.randint(0, NUM_CLASSES, (4, 8, 8))

    matrix, _ = predict(Constant(3), loader(images, targets), torch.device("cpu"))

    expected = ConfusionMatrix()
    expected.update(targets, torch.full_like(targets, 3))
    assert torch.equal(matrix.matrix, expected.matrix)


def test_predict_keeps_only_the_requested_number_of_examples() -> None:
    """Overlays need a handful of images; holding the split would exhaust memory."""
    images = torch.randn(6, 3, 8, 8)
    targets = torch.randint(0, NUM_CLASSES, (6, 8, 8))

    _, examples = predict(Constant(0), loader(images, targets), torch.device("cpu"), keep=3)

    assert examples["images"].shape[0] == 3
    assert examples["targets"].shape[0] == 3
    assert examples["predictions"].shape[0] == 3


def test_predict_keeps_nothing_by_default() -> None:
    images = torch.randn(4, 3, 8, 8)
    targets = torch.randint(0, NUM_CLASSES, (4, 8, 8))

    _, examples = predict(Constant(0), loader(images, targets), torch.device("cpu"))

    assert examples == {}
