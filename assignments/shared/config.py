"""Configuration files merged with `base.yaml` and overridden from the command line."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import io as shared_io


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
