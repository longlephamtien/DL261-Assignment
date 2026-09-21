"""Training entry point, shared by every model.

    python -m src.train --config configs/linear.yaml
    python -m src.train --config configs/mlp.yaml --set seed=1

Implemented by #16.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def fit(config: dict) -> Path:
    """Train one model, write the run directory, and return its path.

    Appends an `EpochRecord` to `history.json` after every epoch, keeps the checkpoint
    selected by `train.checkpoint_metric`, and writes `summary.json` at the end. The loop
    contains no model-specific branches. Move the model and every batch to
    `interfaces.select_device()`; never hardcode `"cuda"` or `"mps"`. Issue #16.
    """
    raise NotImplementedError("issue #16")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one Assignment 1 model.")
    parser.add_argument("--config", type=Path, required=True, help="path to a model config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override; repeatable")
    args = parser.parse_args(argv)

    from .utils import load_config, resolve_path, set_seed

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
