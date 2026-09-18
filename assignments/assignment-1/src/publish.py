"""Upload a finished run to the Hugging Face Hub.

    python -m src.publish results/runs/<run_id>

Credentials come from the repository `.env`. See ../shared/README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shared import hub

ASSIGNMENT = "assignment-1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload one run directory to the Hugging Face Hub.")
    parser.add_argument("run_dir", type=Path, help="path under results/runs/")
    parser.add_argument("--repo-id", default=None, help="defaults to HF_REPO_ID from .env")
    args = parser.parse_args(argv)

    revision = hub.upload_run(args.run_dir, ASSIGNMENT, args.repo_id)
    print(revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
