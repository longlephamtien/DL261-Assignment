"""Upload finished runs to the Hugging Face Hub.

    python -m src.publish <run_id> [<run_id> ...] [--dry-run] [--message "why"]

One Hub commit per run. Credentials come from the repository `.env`. See ../shared/README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from shared import hub, io as shared_io

from .interfaces import EVALUATION_SPLITS
from .utils import evaluation_dir, run_dir as resolve_run

ASSIGNMENT = "assignment-3"


def evaluation_artifacts(run_dir: Path) -> dict[str, Path]:
    config = shared_io.read_yaml(run_dir / "config.yaml")
    return {
        f"evaluation/{split}/{path.name}": path
        for split in EVALUATION_SPLITS
        for path in sorted(evaluation_dir(config, run_dir.name, split).glob("*"))
        if path.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload run directories to the Hugging Face Hub.")
    parser.add_argument("runs", nargs="+", type=resolve_run, help="run ids, or paths to the run directories")
    parser.add_argument("--repo-id", default=None, help="defaults to HF_REPO_ID from .env")
    parser.add_argument("--message", default=None, help="Hub commit message; defaults to naming the run")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded, upload nothing")
    args = parser.parse_args(argv)

    # Every run is checked before anything is uploaded, so a missing report cannot leave the Hub
    # holding half a comparison.
    pending = []
    for run_dir in args.runs:
        extras = evaluation_artifacts(run_dir)
        if not extras:
            parser.error(f"{run_dir.name} has no evaluation report; run python -m src.evaluate first")
        pending.append((run_dir, extras))

    for run_dir, extras in pending:
        if args.dry_run:
            print(run_dir.name)
            for name, path in hub.artifacts(run_dir, extras).items():
                print(f"  {'ok' if path.is_file() else 'MISSING':<8}{name}")
            continue
        revision = hub.upload_run(run_dir, ASSIGNMENT, args.repo_id, message=args.message, extras=extras)
        print(f"{run_dir.name}  {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
