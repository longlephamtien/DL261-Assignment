"""The shared Trainer: what one fit must write, stop on, and keep.

Exercised from Assignment 2 because that is the first consumer; nothing here is segmentation
specific, which is the property the shared package depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shared import Trainer


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 2)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.layer(batch)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A run directory holding only what the trainer reads from it."""
    (tmp_path / "environment.json").write_text(json.dumps({"commit": "abc1234", "hardware": "test"}))
    return tmp_path


def loader() -> DataLoader:
    return DataLoader(TensorDataset(torch.randn(4, 4), torch.randint(0, 2, (4,))), batch_size=2)


def trainer(run_dir: Path, **kwargs) -> Trainer:
    model = Tiny()
    return Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        device=torch.device("cpu"),
        run_dir=run_dir,
        **kwargs,
    )


def scoring(sequence: list[float]):
    """A pass whose checkpoint metric follows `sequence`, so stopping can be driven exactly."""
    calls = {"n": 0}

    def run_epoch_fn(model, loader, criterion, device, optimizer):
        if optimizer is None:  # the validation pass decides the checkpoint
            value = sequence[min(calls["n"], len(sequence) - 1)]
            calls["n"] += 1
            return {"loss": 1.0, "score": value}
        optimizer.step()  # the real pass steps before the scheduler does; keep that order here
        return {"loss": 1.0, "score": 0.0}

    return run_epoch_fn


def test_history_is_written_after_every_epoch(run_dir: Path) -> None:
    """A run killed part way through must still leave what it had measured."""
    coach = trainer(run_dir, checkpoint_metric="val_score")
    coach.fit(3, loader(), loader(), scoring([0.1, 0.2, 0.3]))

    history = json.loads((run_dir / "history.json").read_text())
    assert [record["epoch"] for record in history] == [1, 2, 3]


def test_metric_names_are_prefixed_not_hardcoded(run_dir: Path) -> None:
    """The trainer must carry whatever a pass reports, so other tasks can use it."""
    coach = trainer(run_dir, checkpoint_metric="val_score")
    coach.fit(1, loader(), loader(), scoring([0.5]))

    record = json.loads((run_dir / "history.json").read_text())[0]
    assert set(record) == {
        "epoch", "train_loss", "train_score", "val_loss", "val_score", "seconds", "learning_rate"
    }


def test_the_checkpoint_follows_the_best_epoch(run_dir: Path) -> None:
    coach = trainer(run_dir, checkpoint_metric="val_score", early_stopping_patience=0)
    stats = coach.fit(3, loader(), loader(), scoring([0.1, 0.9, 0.4]))

    assert stats["best_epoch"] == 2
    assert stats["checkpoint_value"] == pytest.approx(0.9)
    assert (run_dir / "checkpoint.pt").is_file()


def test_min_mode_prefers_the_lowest_value(run_dir: Path) -> None:
    coach = trainer(run_dir, checkpoint_metric="val_score", checkpoint_mode="min", early_stopping_patience=0)
    stats = coach.fit(3, loader(), loader(), scoring([0.9, 0.2, 0.7]))

    assert stats["best_epoch"] == 2
    assert stats["checkpoint_value"] == pytest.approx(0.2)


def test_early_stopping_fires_after_the_patience(run_dir: Path) -> None:
    coach = trainer(run_dir, checkpoint_metric="val_score", early_stopping_patience=2)
    stats = coach.fit(10, loader(), loader(), scoring([0.5, 0.1, 0.1, 0.1]))

    assert stats["epochs_trained"] == 3, "one improving epoch, then two without"
    assert stats["best_epoch"] == 1


def test_patience_of_zero_disables_early_stopping(run_dir: Path) -> None:
    coach = trainer(run_dir, checkpoint_metric="val_score", early_stopping_patience=0)
    stats = coach.fit(4, loader(), loader(), scoring([0.5, 0.1, 0.1, 0.1]))

    assert stats["epochs_trained"] == 4


def test_a_missing_checkpoint_metric_is_reported_with_what_was_available(run_dir: Path) -> None:
    """Silently never checkpointing would leave a run with no weights to evaluate."""
    coach = trainer(run_dir, checkpoint_metric="val_miou")

    with pytest.raises(KeyError, match="val_miou"):
        coach.fit(1, loader(), loader(), scoring([0.5]))


def test_an_invalid_checkpoint_mode_is_rejected_at_construction(run_dir: Path) -> None:
    with pytest.raises(ValueError, match="checkpoint_mode"):
        trainer(run_dir, checkpoint_mode="highest")


def test_the_summary_carries_the_run_environment(run_dir: Path) -> None:
    """Section 4.2 requires every result to trace to the commit that produced it."""
    coach = trainer(run_dir, checkpoint_metric="val_score")
    stats = coach.fit(1, loader(), loader(), scoring([0.5]))

    assert stats["commit"] == "abc1234"
    assert stats["hardware"] == "test"
    assert stats["training_seconds"] > 0


def test_the_scheduler_steps_once_per_epoch(run_dir: Path) -> None:
    model = Tiny()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    coach = Trainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        run_dir=run_dir,
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5),
        checkpoint_metric="val_score",
        early_stopping_patience=0,
    )
    coach.fit(3, loader(), loader(), scoring([0.1, 0.2, 0.3]))

    rates = [record["learning_rate"] for record in coach.history]
    assert rates == pytest.approx([1.0, 0.5, 0.25])
