"""Training entry point, shared by every model.

    python -m src.train --config configs/image-only.yaml
    python -m src.train --config configs/improved-fusion.yaml --set seed=1

One pass serves all four models of section 29.1, because each takes the same three arguments and
ignores whichever one it does not read. Implemented by #100, #101 and #102.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from shared import Trainer
from shared import io as shared_io
from shared.runs import set_seed

from .data import build_dataloaders
from .interfaces import NUM_CLASSES, RunSummary, count_parameters, select_device
from .metrics import PairMetrics
from .models import build_model
from .utils import RUNS_ROOT, create_run_dir, load_config, resolve_path


class DiceLoss(nn.Module):
    """Soft Dice on the foreground channel.

    Targets cover 8% of a frame at the median, so cross-entropy alone is dominated by background.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probabilities = logits.softmax(dim=1)[:, 1]
        truth = (targets == 1).to(probabilities.dtype)
        axes = tuple(range(1, probabilities.dim()))
        intersection = (probabilities * truth).sum(dim=axes)
        total = probabilities.sum(dim=axes) + truth.sum(dim=axes)
        return 1.0 - ((2 * intersection + self.smooth) / (total + self.smooth)).mean()


class CombinedLoss(nn.Module):
    """Cross-entropy plus Dice, the usual pairing for a single foreground class."""

    def __init__(self, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.cross_entropy(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_criterion(config: dict) -> nn.Module:
    """Selected by `train.loss`."""
    name = config.get("train", {}).get("loss", "combined").lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return DiceLoss()
    if name == "combined":
        return CombinedLoss(float(config.get("train", {}).get("dice_weight", 1.0)))
    raise ValueError(f"unsupported loss: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Mean loss, mIoU and Dice over one pass; passing an optimizer trains.

    One function serves both passes, so training and validation numbers cannot drift apart
    through a change applied to only one of them.
    """
    training = optimizer is not None
    model.train(training)
    metrics = PairMetrics()
    total_loss = 0.0
    pairs_seen = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["images"].to(device)
            tokens = batch["tokens"].to(device)
            attention = batch["attention"].to(device)
            targets = batch["masks"].to(device)

            logits = model(images, tokens, attention)
            loss = criterion(logits, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.shape[0]
            pairs_seen += images.shape[0]
            metrics.update(targets.cpu(), logits.argmax(dim=1).cpu())

    return {
        "loss": total_loss / pairs_seen,
        "miou": metrics.mean_iou(),
        "dice": metrics.mean_dice(),
    }


def build_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    """AdamW or SGD over the parameters that require gradients."""
    settings = config.get("train", {})
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = float(settings.get("lr", 3e-4))
    weight_decay = float(settings.get("weight_decay", 0.05))

    name = settings.get("optimizer", "adamw").lower()
    if name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(trainable, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"unsupported optimizer: {name}")


def fit(config: dict) -> Path:
    """Train one model, write the run directory, and return its path."""
    set_seed(config.get("seed", 0))
    device = select_device()
    loaders = build_dataloaders(config)
    model = build_model(config["model"]["name"], **config["model"].get("args", {})).to(device)
    run_dir = create_run_dir(config, resolve_path(config.get("output_dir", RUNS_ROOT)))

    settings = config.get("train", {})
    epochs = int(settings.get("epochs", 20))
    optimizer = build_optimizer(config, model)

    scheduler_name = settings.get("scheduler", "cosine").lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"unsupported scheduler: {scheduler_name}")

    trainer = Trainer(
        model=model,
        criterion=build_criterion(config),
        optimizer=optimizer,
        device=device,
        run_dir=run_dir,
        scheduler=scheduler,
        checkpoint_metric=settings.get("checkpoint_metric", "val_miou"),
        checkpoint_mode=settings.get("checkpoint_mode", "max"),
        early_stopping_patience=int(settings.get("early_stopping_patience", 5)),
    )

    stats = trainer.fit(
        epochs=epochs,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        run_epoch_fn=run_epoch,
    )

    arguments = config["model"].get("args", {})
    summary: RunSummary = {
        "run_id": run_dir.name,
        "model": config["model"]["name"],
        "modalities": list(arguments.get("modalities", ["image", "text"])),
        "fusion": arguments.get("fusion", ""),
        "seed": config["seed"],
        "parameters": count_parameters(model),
        **stats,
    }
    shared_io.write_json(run_dir / "summary.json", summary)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one Assignment 3 model.")
    parser.add_argument("--config", type=Path, required=True, help="path to a model config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="config override; repeatable")
    args = parser.parse_args(argv)

    overrides = dict(item.split("=", 1) for item in args.set)
    config = load_config(resolve_path(args.config), overrides)
    print(fit(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
