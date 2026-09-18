"""Configuration, seeding, and run tracking.

Implemented by #12.
"""

from __future__ import annotations

from pathlib import Path


def load_config(path: Path, overrides: dict | None = None) -> dict:
    """Merge `configs/base.yaml` with a per-model file and command-line overrides. Issue #12."""
    raise NotImplementedError("issue #12")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch, including CUDA and cuDNN determinism. Issue #12."""
    raise NotImplementedError("issue #12")


def create_run_dir(config: dict, root: Path) -> Path:
    """Create `results/runs/<run_id>/` and write the resolved config, commit, and versions.

    The directory holds `config.yaml`, `environment.json`, `history.json`, `summary.json`,
    and `checkpoint.pt`. Issue #12.
    """
    raise NotImplementedError("issue #12")
