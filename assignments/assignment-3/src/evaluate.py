"""Score one run on a split and write its report and overlays. Implemented by #156 and #157.

    python -m src.evaluate <run_id> [--split testA]
"""

from __future__ import annotations

import argparse

from .interfaces import EVALUATION_SPLITS, EvaluationReport


def evaluate(run: str, split: str = "val") -> EvaluationReport:
    """Score one run, writing the report and the qualitative examples beside it."""
    raise NotImplementedError("#156 implements the evaluation entry point")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score one run and write its evaluation report.")
    parser.add_argument("run", help="run id, or a path to the run directory")
    parser.add_argument("--split", choices=["val", *EVALUATION_SPLITS], default="val")
    parser.parse_args(argv)
    raise NotImplementedError("#156 implements the evaluation entry point")


if __name__ == "__main__":
    raise SystemExit(main())
