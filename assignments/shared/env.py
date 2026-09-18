"""Environment variables loaded from the repository `.env`.

`.env` is gitignored and holds the Hugging Face repository id and access token. Copy
`.env.example` to `.env` and fill it in. Existing environment variables always win, so CI or
a shell export overrides the file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ID = "HF_REPO_ID"
TOKEN = "HF_TOKEN"


def find_env(start: Path | None = None) -> Path | None:
    """Nearest `.env` searching upwards from `start`, defaulting to the working directory."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load(start: Path | None = None) -> Path | None:
    """Load the nearest `.env` without overriding variables already set."""
    path = find_env(start)
    if path is not None:
        load_dotenv(path, override=False)
    return path


def repo_id(configured: str | None = None) -> str:
    """Repository id from the config, else `HF_REPO_ID`."""
    value = configured or os.environ.get(REPO_ID)
    if not value:
        raise RuntimeError(f"set hub.repo_id in the config, or {REPO_ID} in .env")
    return value


def token() -> str | None:
    """Access token, or None to fall back to the huggingface-cli login cache."""
    return os.environ.get(TOKEN) or None
