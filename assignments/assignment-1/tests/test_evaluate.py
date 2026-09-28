"""Evaluation contract, checked against a synthetic loader rather than a trained run."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import export_examples, measure_latency, predict, synchronize
from src.interfaces import CLASS_NAMES, IMAGE_SHAPE, NUM_CLASSES, select_device
from src.models import build_model

SAMPLES = 32
BATCH = 8


@pytest.fixture
def loader() -> DataLoader:
    images = torch.randn(SAMPLES, *IMAGE_SHAPE)
    labels = torch.randint(0, NUM_CLASSES, (SAMPLES,))
    return DataLoader(TensorDataset(images, labels), batch_size=BATCH)


@pytest.fixture
def model() -> torch.nn.Module:
    return build_model("linear").to(select_device()).eval()


def test_synchronize_accepts_every_device() -> None:
    for device in (torch.device("cpu"), select_device()):
        synchronize(device)


def test_predict_returns_aligned_columns(model: torch.nn.Module, loader: DataLoader) -> None:
    targets, predictions, confidences, images = predict(model, loader, select_device())
    assert targets.shape == predictions.shape == confidences.shape == (SAMPLES,)
    assert images.shape == (SAMPLES, *IMAGE_SHAPE)
    assert predictions.min() >= 0 and predictions.max() < NUM_CLASSES
    assert ((confidences >= 0) & (confidences <= 1)).all(), "confidences come from a softmax"


def test_latency_is_measured_per_batch_size(model: torch.nn.Module) -> None:
    latency = measure_latency(
        model, select_device(), {"batch_sizes": [1, 8], "warmup_batches": 1, "timed_batches": 3}
    )
    assert set(latency) == {"1", "8"}
    assert all(value > 0 for value in latency.values())


def test_examples_are_exported_for_both_outcomes(tmp_path: Path) -> None:
    images = torch.rand(6, *IMAGE_SHAPE)
    targets = torch.arange(6) % NUM_CLASSES
    predictions = targets.clone()
    predictions[3:] = (predictions[3:] + 1) % NUM_CLASSES
    confidences = torch.linspace(0.5, 1.0, 6)

    export_examples(images, targets, predictions, confidences, {"mean": 0.3, "std": 0.35}, 2, tmp_path)
    assert (tmp_path / "correct.png").is_file()
    assert (tmp_path / "incorrect.png").is_file()


def test_errors_are_ordered_by_confidence(tmp_path: Path) -> None:
    """The failure taxonomy needs the most confident mistakes, not the first ones."""
    images = torch.rand(4, *IMAGE_SHAPE)
    targets = torch.zeros(4, dtype=torch.long)
    predictions = torch.ones(4, dtype=torch.long)
    confidences = torch.tensor([0.30, 0.99, 0.40, 0.50])

    export_examples(images, targets, predictions, confidences, {"mean": 0.3, "std": 0.35}, 1, tmp_path)
    assert (tmp_path / "incorrect.png").is_file()
    assert not (tmp_path / "correct.png").is_file(), "no correct prediction to show"


def test_class_names_cover_every_label() -> None:
    assert len(CLASS_NAMES) == NUM_CLASSES
