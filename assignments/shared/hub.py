"""Checkpoint storage on the Hugging Face Hub, shared by all three assignments.

Weights are never committed to git. A run writes `results/runs/<run_id>/` locally, and the
selected runs are uploaded to one Hub repository that holds all three assignments:

    <repo_id>/
      assignment-1/runs/<run_id>/{checkpoint.pt, config.yaml, environment.json, history.json, summary.json}
      assignment-2/runs/<run_id>/...
      assignment-3/runs/<run_id>/...

"""

from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from . import env, io as shared_io

ASSIGNMENTS = ("assignment-1", "assignment-2", "assignment-3")

CHECKPOINT_NAME = "checkpoint.pt"
CHECKSUM_NAME = "checksums.json"
RUN_FILES = (CHECKPOINT_NAME, "config.yaml", "environment.json", "history.json", "summary.json")


def run_path(assignment: str, run_id: str) -> str:
    """Path of one run inside the shared repository."""
    if assignment not in ASSIGNMENTS:
        raise ValueError(f"unknown assignment '{assignment}'; expected one of {', '.join(ASSIGNMENTS)}")
    return f"{assignment}/runs/{run_id}"


def sha256(path: Path) -> str:
    """Checksum published next to every uploaded artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checksums(run_dir: Path) -> dict[str, str]:
    """Checksums of the run files present in `run_dir`, written to the report and the Hub."""
    return {name: sha256(run_dir / name) for name in RUN_FILES if (run_dir / name).is_file()}


def upload_run(
    run_dir: Path,
    assignment: str,
    repo_id: str | None = None,
    private: bool = False,
    message: str | None = None,
) -> str:
    """Upload one run directory with its checksums and return the commit revision to cite."""
    run_dir = Path(run_dir)
    missing = [name for name in RUN_FILES if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing {', '.join(missing)}")

    env.load(run_dir)
    repo_id = env.repo_id(repo_id)
    token = env.token()
    api = HfApi(token=token)

    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    shared_io.write_json(run_dir / CHECKSUM_NAME, checksums(run_dir))
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(run_dir),
        path_in_repo=run_path(assignment, run_dir.name),
        allow_patterns=[*RUN_FILES, CHECKSUM_NAME],
        commit_message=message or f"Add {assignment} run {run_dir.name}",
    )
    return commit.oid


def download_checkpoint(assignment: str, run_id: str, repo_id: str | None = None, revision: str = "main") -> Path:
    """Fetch one checkpoint into the local cache and return its path."""
    env.load()
    return Path(
        hf_hub_download(
            repo_id=env.repo_id(repo_id),
            filename=f"{run_path(assignment, run_id)}/{CHECKPOINT_NAME}",
            revision=revision,
            token=env.token(),
        )
    )
