"""Text file IO with one encoding on every platform.

Python reads and writes text in the locale encoding, which is UTF-8 on macOS and Linux but
often cp1252 on Windows, and it translates newlines on write. Every config, split, history,
and summary goes through this module, so a file produced on one platform loads unchanged on
another and diffs stay clean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ENCODING = "utf-8"
NEWLINE = "\n"


def read_text(path: Path) -> str:
    return Path(path).read_text(encoding=ENCODING)


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=ENCODING, newline=NEWLINE) as handle:
        handle.write(text)


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + NEWLINE)


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(read_text(path))


def write_yaml(path: Path, data: Any) -> None:
    write_text(path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
