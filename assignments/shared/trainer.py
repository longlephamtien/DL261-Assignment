"""The training loop shared by every assignment: epochs, checkpointing, and early stopping.

What differs between assignments is the metric a pass reports, not the loop around it. A pass
therefore returns a mapping of metric name to value, and the trainer prefixes those names with
`train_` and `val_` rather than knowing any of them. Nothing here mentions a task.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import io as shared_io

EpochFunction = Callable[
    [nn.Module, DataLoader, nn.Module, torch.device, torch.optim.Optimizer | None],
    dict[str, float],
]
"""One pass over a loader. Training when given an optimizer, scoring when given None."""


class Trainer:
    """Runs the epochs of one model and keeps the run directory current as it goes.

    History is written after every epoch rather than at the end, so a run killed part way
    through still leaves behind what it had measured.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        run_dir: Path,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        checkpoint_metric: str = "val_miou",
        checkpoint_mode: str = "max",
        early_stopping_patience: int = 8,
    ) -> None:
        if checkpoint_mode not in ("max", "min"):
            raise ValueError(f"checkpoint_mode must be 'max' or 'min', not '{checkpoint_mode}'")

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.run_dir = Path(run_dir)
        self.scheduler = scheduler
        self.checkpoint_metric = checkpoint_metric
        self.checkpoint_mode = checkpoint_mode
        self.patience = early_stopping_patience

        self.best = -float("inf") if checkpoint_mode == "max" else float("inf")
        self.best_epoch = 0
        self.waited = 0
        self.history: list[dict[str, Any]] = []

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_epoch_fn: EpochFunction,
    ) -> dict[str, Any]:
        """Train for up to `epochs`, and return the fields a run summary needs."""
        started = time.time()
        environment = shared_io.read_json(self.run_dir / "environment.json")

        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            learning_rate = float(self.optimizer.param_groups[0]["lr"])

            train = run_epoch_fn(self.model, train_loader, self.criterion, self.device, self.optimizer)
            validation = run_epoch_fn(self.model, val_loader, self.criterion, self.device, None)
            if self.scheduler is not None:
                self.scheduler.step()

            record: dict[str, Any] = {"epoch": epoch}
            record.update({f"train_{name}": value for name, value in train.items()})
            record.update({f"val_{name}": value for name, value in validation.items()})
            record["seconds"] = time.time() - epoch_started
            record["learning_rate"] = learning_rate

            self.history.append(record)
            shared_io.write_json(self.run_dir / "history.json", self.history)
            print(self._describe(record, epochs))

            if self.checkpoint_metric not in record:
                raise KeyError(
                    f"checkpoint metric '{self.checkpoint_metric}' is not among "
                    f"{sorted(record)}; a pass must report it under that name"
                )

            if self._improved(record[self.checkpoint_metric]):
                self.best = record[self.checkpoint_metric]
                self.best_epoch, self.waited = epoch, 0
                torch.save(self.model.state_dict(), self.run_dir / "checkpoint.pt")
            else:
                self.waited += 1
                if self.patience > 0 and self.waited >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return {
            "epochs_trained": len(self.history),
            "best_epoch": self.best_epoch,
            "checkpoint_metric": self.checkpoint_metric,
            "checkpoint_value": float(self.best),
            "training_seconds": time.time() - started,
            "commit": environment.get("commit", "unknown"),
            "hardware": environment.get("hardware", "unknown"),
        }

    def _improved(self, value: float) -> bool:
        return value > self.best if self.checkpoint_mode == "max" else value < self.best

    def _describe(self, record: dict[str, Any], epochs: int) -> str:
        """One progress line, naming whatever metrics the pass reported."""
        metrics = " ".join(
            f"{name} {value:.4f}"
            for name, value in record.items()
            if name not in ("epoch", "seconds", "learning_rate")
        )
        return f"Epoch {record['epoch']:02d}/{epochs:02d} | {metrics} | {record['seconds']:.1f}s"
