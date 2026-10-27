"""Training entry point, shared by every model.

    python -m src.train --config configs/linear.yaml
    python -m src.train --config configs/mlp.yaml --set seed=1

Implemented by #16.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from shared import Trainer, io as shared_io

from .data import build_dataloaders
from .figures import label, plot_curves
from .interfaces import RunSummary, count_parameters, select_device
from .models import build_model
from .utils import RUNS_ROOT, create_run_dir, load_config, resolve_path, set_seed


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Mean loss, accuracy, and macro-F1 over one pass; passing an optimizer trains.

    The names are what `Trainer` prefixes with `train_` and `val_` to build each history
    record, which is why they are returned as a mapping rather than a tuple.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            predictions.extend(outputs.argmax(dim=1).detach().cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def fit(config: dict) -> Path:
    """Train one model, write the run directory, and return its path.

    Loaders are built before the run directory so the normalization statistics they compute
    are recorded in `config.yaml`.
    """
    device = select_device()
    loaders = build_dataloaders(config)
    model = build_model(config["model"]["name"], **config["model"].get("args", {})).to(device)

    output_root = resolve_path(config.get("output_dir", RUNS_ROOT))
    run_dir = create_run_dir(config, output_root)

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
    elif sched_name == "none":
        scheduler = None
    else:
        raise ValueError(f"unsupported scheduler: {sched_name}")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        run_dir=run_dir,
        scheduler=scheduler,
        checkpoint_metric=checkpoint_metric,
        checkpoint_mode=checkpoint_mode,
        early_stopping_patience=patience,
    )
    stats = trainer.fit(epochs, loaders["train"], loaders["val"], run_epoch)

    summary: RunSummary = {
        "run_id": run_dir.name,
        "model": config["model"]["name"],
        "representation": config.get("input", {}).get("representation", "image"),
        "seed": config["seed"],
        "parameters": count_parameters(model),
        **stats,
    }
    shared_io.write_json(run_dir / "summary.json", summary)
    plot_curves({label(summary): [trainer.history]}, run_dir / "curves")
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
