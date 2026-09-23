"""Configuration loading, seeding, and per-run directories."""

from __future__ import annotations

import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
import torchvision

from shared import io as shared_io

from .interfaces import describe_hardware, select_device

ASSIGNMENT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path against the assignment folder, so the working directory never matters."""
    return ASSIGNMENT_ROOT / path


def load_config(path: Path, overrides: dict[str, str] | None = None) -> dict:
    """Merge `base.yaml` with a per-model file, then apply dotted `key=value` overrides."""
    path = Path(path)
    config = shared_io.read_yaml(path) or {}

    base_path = path.parent / "base.yaml"
    if base_path.is_file() and base_path != path:
        config = _deep_merge(shared_io.read_yaml(base_path) or {}, config)

    for key, value in (overrides or {}).items():
        _set_dotted(config, key, _coerce(value))
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_dotted(config: dict, dotted_key: str, value: Any) -> None:
    node = config
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def _coerce(value: str) -> Any:
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


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


def create_run_dir(config: dict, root: Path) -> Path:
    """Create `results/runs/<run_id>/` and write config.yaml and environment.json."""
    run_dir = Path(root) / _run_id(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    shared_io.write_yaml(run_dir / "config.yaml", config)
    shared_io.write_json(run_dir / "environment.json", _environment())
    return run_dir


def _run_id(config: dict) -> str:
    model = config.get("model", {}).get("name", "model")
    seed = config.get("seed", 0)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{model}-seed{seed}-{stamp}"


def _environment() -> dict:
    return {
        "commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "hardware": describe_hardware(),
        "torch_threads": torch.get_num_threads(),
    }


def _git_commit() -> str:
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
