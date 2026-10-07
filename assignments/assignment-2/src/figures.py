"""Plots and tables derived from finished runs.

`train.fit` calls `plot_curves` itself, so every run directory already holds its own curves.
This module compares runs against each other:

    python -m src.figures <run_id> [...] [--name baselines] [--split val]

writing one directory per comparison under `results/figures/`. Metrics come from the
evaluation reports, not from `history.json`. Implemented by #73.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shared.figures import COLOURS, save  # noqa: F401  re-exported for the plotting work

from .interfaces import EVALUATION_SPLITS, EpochRecord, RunSummary
from .utils import run_dir as resolve_run

FIGURES = "results/figures"


def load_run(directory: Path) -> tuple[dict, RunSummary, list[EpochRecord]]:
    """Resolved configuration, summary, and per-epoch history of one finished run."""
    raise NotImplementedError("#73 implements the comparison outputs")


def plot_curves(runs: dict[str, list[EpochRecord]], path: Path) -> None:
    """Training and validation loss and mIoU for every run, on shared axes. Implemented by #73."""
    raise NotImplementedError("#73 implements the comparison outputs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export learning curves and the metrics table for finished runs.")
    parser.add_argument("runs", nargs="+", type=resolve_run, help="run ids, or paths to the run directories")
    parser.add_argument("--name", default=None, help="comparison directory; defaults to the models compared")
    parser.add_argument("--split", choices=EVALUATION_SPLITS, default="val", help="which reports to tabulate")
    parser.parse_args(argv)

    raise NotImplementedError("#73 implements the comparison outputs")


if __name__ == "__main__":
    raise SystemExit(main())
