"""Paths, configuration, seeding, and per-run directories for Assignment 3."""

from __future__ import annotations

from pathlib import Path

from shared import config as shared_config, runs as shared_runs

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = "results/runs"


def resolve_path(path: str | Path) -> Path:
    """Resolve against the assignment folder, so commands work from any directory."""
    path = Path(path)
    return path if path.is_absolute() else ASSIGNMENT_ROOT / path


def run_dir(run: str | Path) -> Path:
    """A run id or a path to a run directory, resolved to the directory."""
    candidate = Path(run)
    if candidate.is_dir():
        return candidate
    directory = resolve_path(RUNS_ROOT) / str(run)
    if not directory.is_dir():
        raise FileNotFoundError(f"no run directory for {run}")
    return directory


def load_config(path: Path, overrides: dict[str, str] | None = None) -> dict:
    """Merge base and model configuration, applying dotted overrides."""
    return shared_config.load_config(resolve_path(path), overrides)


def create_run_dir(config: dict, root: Path | None = None) -> Path:
    """Create the run directory with its configuration, environment and commit."""
    return shared_runs.create_run_dir(config, resolve_path(root or RUNS_ROOT))
