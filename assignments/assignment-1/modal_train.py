"""Run Assignment 1 training on Modal cloud GPU without touching the core codebase.

Usage:
    cd assignments/assignment-1
    modal run modal_train.py --config configs/mlp.yaml
    modal run modal_train.py --config configs/linear.yaml
    modal run modal_train.py --config configs/mlp.yaml --set-arg "seed=1"
"""

from __future__ import annotations

import io
import os
import sys
import tarfile
from pathlib import Path

import modal

app = modal.App("dl261-assignment1")

ASSIGNMENT_DIR = Path(__file__).resolve().parent
SHARED_DIR = ASSIGNMENT_DIR.parent / "shared"

# Define remote container environment
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "scikit-learn",
        "pyyaml",
        "python-dotenv",
        "huggingface-hub",
        "tqdm",
    )
    .add_local_dir(
        SHARED_DIR,
        remote_path="/workspace/assignments/shared",
    )
    .add_local_dir(
        ASSIGNMENT_DIR / "configs",
        remote_path="/workspace/assignments/assignment-1/configs",
    )
    .add_local_dir(
        ASSIGNMENT_DIR / "splits",
        remote_path="/workspace/assignments/assignment-1/splits",
    )
    .add_local_dir(
        ASSIGNMENT_DIR / "src",
        remote_path="/workspace/assignments/assignment-1/src",
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
)
def train_remote(config_path: str, overrides_list: list[str]) -> tuple[str, bytes]:
    """Execute training inside the Modal container on GPU and return the zipped run artifacts."""
    work_dir = Path("/workspace/assignments/assignment-1")
    os.chdir(work_dir)

    # Ensure Python can find both 'shared' and local 'src'
    sys.path.insert(0, "/workspace/assignments")
    sys.path.insert(0, "/workspace/assignments/assignment-1")

    from src.data import prepare_dataset
    from src.train import fit
    from src.utils import load_config, resolve_path, set_seed

    overrides = {}
    for item in overrides_list:
        if "=" in item:
            k, v = item.split("=", 1)
            overrides[k] = v

    config = load_config(resolve_path(config_path), overrides)
    set_seed(config["seed"])

    print("Ensuring dataset is ready on remote container...")
    prepare_dataset(config)

    print(f"Training {config['model']['name']} on GPU (seed={config['seed']})...")
    run_dir = fit(config)
    print(f"Training completed. Run directory created: {run_dir.name}")

    # Pack the generated run_dir into an in-memory tarball
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(run_dir, arcname=run_dir.name)
    buf.seek(0)

    return run_dir.name, buf.getvalue()


@app.local_entrypoint()
def main(config: str = "configs/mlp.yaml", set_arg: str = ""):
    """Local CLI wrapper that triggers remote training and automatically downloads the results."""
    overrides = [set_arg] if set_arg else []
    print(f"Submitting job to Modal GPU (T4) using config: {config}")

    run_name, tar_bytes = train_remote.remote(config, overrides)

    # Extract remote results directly into local results/runs/
    local_runs_dir = ASSIGNMENT_DIR / "results" / "runs"
    local_runs_dir.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO(tar_bytes)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(path=local_runs_dir)

    saved_path = local_runs_dir / run_name
    print(f"Successfully downloaded run artifacts to local path:")
    print(f"   -> {saved_path}")
