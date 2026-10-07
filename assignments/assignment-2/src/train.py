"""Training entry point, shared by every model.

    python -m src.train --config configs/unet.yaml
    python -m src.train --config configs/deeplabv3.yaml --set seed=1

Implemented by #65 and #66.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .interfaces import select_device
from .utils import load_config, resolve_path, set_seed


def build_criterion(config: dict) -> nn.Module:
    """Cross-entropy or Dice, selected by `train.loss`. Implemented by #65."""
    raise NotImplementedError("#65 implements the losses")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Mean loss and mIoU over one pass; passing an optimizer trains. Implemented by #65."""
    raise NotImplementedError("#65 implements the training loop")


def fit(config: dict) -> Path:
    """Train one model and write its run directory. Implemented by #65."""
    raise NotImplementedError("#65 implements the training loop")


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
