"""Paths, configuration, seeding, and per-run directories for Assignment 1."""

from __future__ import annotations

from pathlib import Path

from shared.config import load_config as _load_config
from shared.runs import create_run_dir as _create_run_dir, set_seed  # noqa: F401  re-exported

from .interfaces import Split

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = "results/runs"


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the assignment folder, so the working directory never matters."""
    return ASSIGNMENT_ROOT / path


def run_dir(run: str | Path) -> Path:
    """A run directory, named either by its run id or by a path to it."""
    for candidate in (resolve_path(run), resolve_path(Path(RUNS_ROOT) / run)):
        if candidate.is_dir():
            return candidate
    raise ValueError(f"no run '{run}' under {RUNS_ROOT}/")


def evaluation_dir(config: dict, run_id: str, split: Split) -> Path:
    """Report and example grids of one run and split; also the layout published to the Hub."""
    return resolve_path(config["evaluation"]["output_dir"]) / run_id / split


def report_path(config: dict, run_id: str, split: Split) -> Path:
    return evaluation_dir(config, run_id, split) / "report.json"


def load_config(path: Path, overrides: dict[str, str] | None = None) -> dict:
    """Merge `base.yaml` with a per-model file, then apply dotted `key=value` overrides."""
    return _load_config(path, overrides)


def create_run_dir(config: dict, root: Path) -> Path:
    """Create `results/runs/<run_id>/` and write config.yaml and environment.json."""
    return _create_run_dir(config, root)
