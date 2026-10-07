"""Metrics, latency, and prediction overlays for one finished run.

    python -m src.evaluate <run_id> [--split val]

Validation is the default because Cityscapes withholds the test labels. Output goes to
`results/evaluation/<run_id>/<split>/` and is published with the run. Implemented by #63.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .interfaces import EVALUATION_SPLITS, EvaluationReport, RunSummary, Split
from .utils import run_dir as resolve_run


def load_checkpoint(directory: Path) -> tuple[dict, RunSummary, nn.Module]:
    """Rebuild the model of a run and restore its selected weights. Implemented by #63."""
    raise NotImplementedError("#63 implements the evaluation suite")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Tensor, Tensor]:
    """Accumulated confusion matrix and a handful of images kept for overlays. Implemented by #63."""
    raise NotImplementedError("#63 implements the evaluation suite")


def export_overlays(images: Tensor, targets: Tensor, predictions: Tensor, count: int, directory: Path) -> None:
    """Predicted masks drawn over their images, best and worst by image mIoU. Implemented by #74."""
    raise NotImplementedError("#74 implements the qualitative evaluation")


def evaluate(run: str | Path, split: Split = "val") -> EvaluationReport:
    """Score one run on a split and write its report and overlays. Implemented by #63."""
    raise NotImplementedError("#63 implements the evaluation suite")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score one run and write its evaluation report.")
    parser.add_argument("run", type=resolve_run, help="run id, or a path to the run directory")
    parser.add_argument("--split", choices=EVALUATION_SPLITS, default="val", help="test only after model selection")
    args = parser.parse_args(argv)

    report = evaluate(args.run, args.split)
    print(f"{report['model']:<12} {args.split:<5} mIoU {report['miou']:.4f}  Dice {report['dice']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
