"""Configuration loading and CLI override utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw)!r}")
    return raw


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _set_nested(config: dict[str, Any], key_path: list[str], value: Any) -> None:
    node: dict[str, Any] = config
    for key in key_path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[key_path[-1]] = value


def apply_cli_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = dict(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be in key=value format, got: {item}")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError(f"Override key cannot be empty: {item}")
        _set_nested(out, key.split("."), _coerce_value(raw_value))
    return out
