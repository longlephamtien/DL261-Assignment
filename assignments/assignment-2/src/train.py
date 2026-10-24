"""Training entry point, shared by every model.

    python -m src.train --config configs/unet.yaml
    python -m src.train --config configs/deeplabv3.yaml --set seed=1

Implemented by #65 and #66.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from shared import io as shared_io

from .data import build_dataloaders
from .interfaces import IGNORE_INDEX, NUM_CLASSES, EpochRecord, RunSummary, count_parameters, select_device
from .metrics import ConfusionMatrix
from .models import build_model
from .utils import RUNS_ROOT, create_run_dir, load_config, resolve_path, set_seed


class DiceLoss(nn.Module):
    """Soft Dice over the predicted probabilities, ignoring unlabelled pixels.

    Cross-entropy weights every pixel equally, so the classes that cover most of a street scene
    dominate it. Dice is computed per class and averaged, which gives a thin class like a pole
    the same say as the road.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        keep = targets != IGNORE_INDEX
        if not keep.any():
            return logits.sum() * 0.0

        probabilities = logits.softmax(dim=1).permute(0, 2, 3, 1)[keep]
        one_hot = nn.functional.one_hot(targets[keep], NUM_CLASSES).to(probabilities.dtype)

        intersection = (probabilities * one_hot).sum(dim=0)
        total = probabilities.sum(dim=0) + one_hot.sum(dim=0)
        return 1.0 - ((2 * intersection + self.smooth) / (total + self.smooth)).mean()


def build_criterion(config: dict) -> nn.Module:
    """Cross-entropy or Dice, selected by `train.loss`."""
    name = config.get("train", {}).get("loss", "cross_entropy").lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    if name == "dice":
        return DiceLoss()
    raise ValueError(f"unsupported loss: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    """Mean loss, mIoU and Dice over one pass; passing an optimizer trains.

    One function for both passes, so the training and validation numbers can never drift apart
    through a change applied to only one of them.
    """
    training = optimizer is not None
    model.train(training)
    matrix = ConfusionMatrix()
    total_loss = 0.0
    images_seen = 0

    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.shape[0]
            images_seen += images.shape[0]
            matrix.update(targets.cpu(), logits.argmax(dim=1).cpu())

    return total_loss / images_seen, matrix.mean_iou(), matrix.mean_dice()


def fit(config: dict) -> Path:
    """Train one model, write the run directory, and return its path."""
    device = select_device()
    loaders = build_dataloaders(config)
    model = build_model(config["model"]["name"], **config["model"].get("args", {})).to(device)

    run_dir = create_run_dir(config, resolve_path(config.get("output_dir", RUNS_ROOT)))
    env = shared_io.read_json(run_dir / "environment.json")

    settings = config.get("train", {})
    epochs = int(settings.get("epochs", 40))
    patience = int(settings.get("early_stopping_patience", 8))
    checkpoint_metric = settings.get("checkpoint_metric", "val_miou")
    checkpoint_mode = settings.get("checkpoint_mode", "max")

    criterion = build_criterion(config)
    trainable = [p for p in model.parameters() if p.requires_grad]

    optimizer_name = settings.get("optimizer", "adamw").lower()
    lr, weight_decay = float(settings.get("lr", 3e-4)), float(settings.get("weight_decay", 0.05))
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(trainable, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")

    scheduler_name = settings.get("scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"unsupported scheduler: {scheduler_name}")

    history: list[EpochRecord] = []
    best = -float("inf") if checkpoint_mode == "max" else float("inf")
    best_epoch, waited = 0, 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, train_miou, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_miou, val_dice = run_epoch(model, loaders["val"], criterion, device)
        if scheduler is not None:
            scheduler.step()

        record: EpochRecord = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "val_dice": val_dice,
            "seconds": time.time() - epoch_started,
            "learning_rate": current_lr,
        }
        history.append(record)
        shared_io.write_json(run_dir / "history.json", history)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | train loss {train_loss:.4f} mIoU {train_miou:.4f} | "
            f"val loss {val_loss:.4f} mIoU {val_miou:.4f} Dice {val_dice:.4f} | {record['seconds']:.1f}s"
        )

        value = record[checkpoint_metric]
        improved = value > best if checkpoint_mode == "max" else value < best
        if improved:
            best, best_epoch, waited = value, epoch, 0
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        else:
            waited += 1
            if patience > 0 and waited >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    summary: RunSummary = {
        "run_id": run_dir.name,
        "model": config["model"]["name"],
        "backbone": config["model"].get("args", {}).get("backbone", ""),
        "seed": config["seed"],
        "parameters": count_parameters(model),
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_value": float(best),
        "training_seconds": time.time() - started,
        "commit": env.get("commit", "unknown"),
        "hardware": env.get("hardware", "unknown"),
    }
    shared_io.write_json(run_dir / "summary.json", summary)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one Assignment 2 model.")
    parser.add_argument("--config", type=Path, required=True, help="path to a model config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override; repeatable")
    args = parser.parse_args(argv)

    overrides = dict(item.split("=", 1) for item in args.set)
    config = load_config(resolve_path(args.config), overrides)
    if not config.get("model", {}).get("name"):
        parser.error(f"{args.config} declares no model.name; pass a model config such as configs/unet.yaml")
    set_seed(config["seed"])
    print(fit(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
