"""Checkpoint storage on the Hugging Face Hub, shared by all three assignments.

Weights are never committed to git. A run writes `results/runs/<run_id>/` locally, and the
selected runs are uploaded to one Hub repository that holds all three assignments:

    <repo_id>/
      assignment-1/runs/<run_id>/{checkpoint.pt, config.yaml, environment.json, history.json, summary.json}
      assignment-1/runs/<run_id>/evaluation/<split>/{report.json, correct.png, incorrect.png}
      assignment-2/runs/<run_id>/...
      assignment-3/runs/<run_id>/...

Evaluation output lives outside the run directory, so each assignment passes it as `extras`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

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


def artifacts(run_dir: Path, extras: dict[str, Path] | None = None) -> dict[str, Path]:
    """Files an upload covers, keyed by their path inside the run's folder on the Hub."""
    return {name: run_dir / name for name in RUN_FILES} | {name: Path(path) for name, path in (extras or {}).items()}


def checksums(files: dict[str, Path]) -> dict[str, str]:
    """Checksums published next to every upload."""
    return {name: sha256(path) for name, path in files.items()}


def upload_run(
    run_dir: Path,
    assignment: str,
    repo_id: str | None = None,
    private: bool = False,
    message: str | None = None,
    extras: dict[str, Path] | None = None,
) -> str:
    """Upload one run with its checksums in a single commit, and return the revision to cite."""
    run_dir = Path(run_dir)
    files = artifacts(run_dir, extras)
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing {', '.join(missing)}")

    env.load(run_dir)
    repo_id = env.repo_id(repo_id)
    token = env.token()
    api = HfApi(token=token)

    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    shared_io.write_json(run_dir / CHECKSUM_NAME, checksums(files))
    files[CHECKSUM_NAME] = run_dir / CHECKSUM_NAME

    prefix = run_path(assignment, run_dir.name)
    commit = api.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=[CommitOperationAdd(f"{prefix}/{name}", str(path)) for name, path in files.items()],
        commit_message=message or f"Add {assignment} run {run_dir.name}",
    )
    return commit.oid


def download_file(
    assignment: str,
    run_id: str,
    name: str = CHECKPOINT_NAME,
    repo_id: str | None = None,
    revision: str = "main",
) -> Path:
    """Fetch one file of a published run into the local cache and return its path."""
    env.load()
    return Path(
        hf_hub_download(
            repo_id=env.repo_id(repo_id),
            filename=f"{run_path(assignment, run_id)}/{name}",
            revision=revision,
            token=env.token(),
        )
    )


def download_checkpoint(assignment: str, run_id: str, repo_id: str | None = None, revision: str = "main") -> Path:
    """Fetch one checkpoint into the local cache and return its path."""
    return download_file(assignment, run_id, CHECKPOINT_NAME, repo_id, revision)
