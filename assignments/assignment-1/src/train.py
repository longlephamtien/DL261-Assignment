"""Training entry point, shared by every model.

    python -m src.train --config configs/linear.yaml
    python -m src.train --config configs/mlp.yaml --set seed=1

Implemented by #16.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from shared import io as shared_io

from .data import build_dataloaders
from .interfaces import EpochRecord, RunSummary, count_parameters, select_device
from .models import build_model
from .utils import create_run_dir, load_config, resolve_path, set_seed


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, float(acc), float(f1)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return avg_loss, float(acc), float(f1)


def fit(config: dict) -> Path:
    """Train one model, write the run directory, and return its path.

    Appends an `EpochRecord` to `history.json` after every epoch, keeps the checkpoint
    selected by `train.checkpoint_metric`, and writes `summary.json` at the end. The loop
    contains no model-specific branches. Move the model and every batch to
    `interfaces.select_device()`; never hardcode `"cuda"` or `"mps"`. Issue #16.
    """
    device = select_device()
    output_root = resolve_path(config.get("output_dir", "results/runs"))
    run_dir = create_run_dir(config, output_root)
    env = shared_io.read_json(run_dir / "environment.json")

    #Tải dữ liệu & khởi tạo model
    loaders = build_dataloaders(config)
    model = build_model(config["model"]["name"], **config["model"].get("args", {})).to(device)


    #Cấu hình hyperparameter
    train_cfg = config.get("train", {})
    epochs = int(train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("lr", 3e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.05))
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    patience = int(train_cfg.get("early_stopping_patience", 5))
    checkpoint_metric = train_cfg.get("checkpoint_metric", "val_macro_f1")
    checkpoint_mode = train_cfg.get("checkpoint_mode", "max")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    opt_name = train_cfg.get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"unsupported optimizer: {opt_name}")

    sched_name = train_cfg.get("scheduler", "cosine").lower()
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    history: list[EpochRecord] = []
    best_val = -float("inf") if checkpoint_mode == "max" else float("inf")
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = float(optimizer.param_groups[0]["lr"])

        train_loss, train_acc, train_f1 = _train_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc, val_f1 = _evaluate(model, loaders["val"], criterion, device)

        if scheduler is not None:
            scheduler.step()

        record: EpochRecord = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "train_macro_f1": round(train_f1, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_f1, 4),
            "seconds": round(time.time() - epoch_start, 2),
            "learning_rate": current_lr,
        }
        history.append(record)
        shared_io.write_json(run_dir / "history.json", history)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train loss: {train_loss:.4f} acc: {train_acc:.4f} f1: {train_f1:.4f} | "
            f"val loss: {val_loss:.4f} acc: {val_acc:.4f} f1: {val_f1:.4f} | "
            f"{record['seconds']}s"
        )

        metric_val = record[checkpoint_metric]
        is_best = (metric_val > best_val) if checkpoint_mode == "max" else (metric_val < best_val)
        if is_best:
            best_val = metric_val
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), run_dir / "checkpoint.pt")
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    summary: RunSummary = {
        "run_id": run_dir.name,
        "model": config["model"]["name"],
        "representation": config.get("input", {}).get("representation", "image"),
        "seed": config["seed"],
        "parameters": count_parameters(model),
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_value": float(best_val),
        "training_seconds": round(time.time() - start_time, 2),
        "commit": env.get("commit", "unknown"),
        "hardware": env.get("hardware", "unknown"),
    }
    shared_io.write_json(run_dir / "summary.json", summary)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one Assignment 1 model.")
    parser.add_argument("--config", type=Path, required=True, help="path to a model config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override; repeatable")
    args = parser.parse_args(argv)

    overrides = dict(item.split("=", 1) for item in args.set)
    config = load_config(resolve_path(args.config), overrides)
    if not config.get("model", {}).get("name"):
        parser.error(f"{args.config} declares no model.name; pass a model config such as configs/linear.yaml")
    set_seed(config["seed"])
    run_dir = fit(config)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
