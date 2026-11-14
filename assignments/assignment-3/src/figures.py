"""Plots and tables derived from finished runs. Implemented by #158.

    python -m src.figures <run_id> [...] [--name all] [--split testA]

The averaging, styling and the decision rule live in `shared.figures`.
"""

from __future__ import annotations

import argparse

from .interfaces import EVALUATION_SPLITS

FIGURES = "results/figures"

PANELS = (("loss", "training objective"), ("miou", "mean IoU"))
"""Learning-curve panels, matching what the trainer records per epoch."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export curves and the metrics table.")
    parser.add_argument("runs", nargs="*", help="run ids, or paths to the run directories")
    parser.add_argument("--name", default=None, help="comparison directory")
    parser.add_argument("--split", choices=["val", *EVALUATION_SPLITS], default="val")
    parser.parse_args(argv)
    raise NotImplementedError("#158 fills the comparison outputs")


if __name__ == "__main__":
    raise SystemExit(main())
