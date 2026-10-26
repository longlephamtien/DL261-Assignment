"""Shared training loop, checkpointing, and run tracking for assignment models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import io as shared_io


EpochResult = Union[Dict[str, float], Tuple[float, ...]]
EpochFunction = Callable[
    [nn.Module, DataLoader, nn.Module, torch.device, Optional[torch.optim.Optimizer]],
    EpochResult,
]


class Trainer:
    """Trainer orchestrating epoch passes, early stopping, and artifact saving."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        run_dir: Path,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        checkpoint_metric: str = "val_miou",
        checkpoint_mode: str = "max",
        early_stopping_patience: int = 8,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.run_dir = Path(run_dir)
        self.scheduler = scheduler
        self.checkpoint_metric = checkpoint_metric
        self.checkpoint_mode = checkpoint_mode.lower()
        self.patience = early_stopping_patience

        self.best = -float("inf") if self.checkpoint_mode == "max" else float("inf")
        self.best_epoch = 0
        self.waited = 0
        self.history: List[Dict[str, Any]] = []

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_epoch_fn: EpochFunction,
    ) -> Dict[str, Any]:
        """Execute the training process and return execution metadata."""
        started = time.time()
        env = shared_io.read_json(self.run_dir / "environment.json")

        for epoch in range(1, epochs + 1):
            epoch_started = time.time()
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            train_res = run_epoch_fn(self.model, train_loader, self.criterion, self.device, self.optimizer)
            val_res = run_epoch_fn(self.model, val_loader, self.criterion, self.device, None)

            if isinstance(train_res, tuple):
                train_loss, train_miou = train_res[0], train_res[1]
                val_loss, val_miou, val_dice = val_res[0], val_res[1], val_res[2]
            else:
                train_loss, train_miou = train_res["loss"], train_res["miou"]
                val_loss, val_miou, val_dice = val_res["loss"], val_res["miou"], val_res["dice"]

            if self.scheduler is not None:
                self.scheduler.step()

            record: Dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_miou": train_miou,
                "val_loss": val_loss,
                "val_miou": val_miou,
                "val_dice": val_dice,
                "seconds": time.time() - epoch_started,
                "learning_rate": current_lr,
            }
            self.history.append(record)
            shared_io.write_json(self.run_dir / "history.json", self.history)

            print(
                f"Epoch {epoch:02d}/{epochs:02d} | train loss {train_loss:.4f} mIoU {train_miou:.4f} | "
                f"val loss {val_loss:.4f} mIoU {val_miou:.4f} Dice {val_dice:.4f} | {record['seconds']:.1f}s"
            )

            value = record.get(self.checkpoint_metric)
            if value is None:
                raise KeyError(
                    f"checkpoint_metric '{self.checkpoint_metric}' is missing from epoch record: {list(record.keys())}"
                )

            improved = value > self.best if self.checkpoint_mode == "max" else value < self.best
            if improved:
                self.best, self.best_epoch, self.waited = value, epoch, 0
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
            "commit": env.get("commit", "unknown"),
            "hardware": env.get("hardware", "unknown"),
        }