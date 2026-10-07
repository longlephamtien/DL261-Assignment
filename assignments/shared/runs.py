"""Seeding and the per-run directory that makes a reported number traceable."""

from __future__ import annotations

import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from . import io as shared_io
from .device import describe_hardware, select_device


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch on whichever device is selected: CUDA, MPS, or CPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = select_device()
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def run_id(config: dict) -> str:
    """`<model>-seed<seed>-<timestamp>`, unique per invocation and sortable by time."""
    model = config.get("model", {}).get("name", "model")
    seed = config.get("seed", 0)
    return f"{model}-seed{seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def create_run_dir(config: dict, root: Path) -> Path:
    """Create `<root>/<run_id>/` and write config.yaml and environment.json."""
    directory = Path(root) / run_id(config)
    directory.mkdir(parents=True, exist_ok=False)
    shared_io.write_yaml(directory / "config.yaml", config)
    shared_io.write_json(directory / "environment.json", environment())
    return directory


def environment() -> dict:
    """Commit, library versions, hardware, and thread count, recorded once per run."""
    versions = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}
    for name in ("torchvision", "sklearn"):
        try:
            versions[name.replace("sklearn", "scikit_learn")] = __import__(name).__version__
        except ImportError:
            pass
    return {"commit": git_commit(), **versions, "hardware": describe_hardware(), "torch_threads": torch.get_num_threads()}


def git_commit() -> str:
    """The commit a run was produced at, or 'unknown' outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
