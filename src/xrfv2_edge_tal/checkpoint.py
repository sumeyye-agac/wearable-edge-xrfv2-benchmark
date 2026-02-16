"""Checkpoint serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

JSON_PREFIX = "json::"


def save_checkpoint(path: str | Path, state_dict: dict[str, Any], metadata: dict[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        f"{JSON_PREFIX}metadata": np.array(json.dumps(metadata), dtype=object),
    }

    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            payload[key] = value
        else:
            payload[f"{JSON_PREFIX}{key}"] = np.array(json.dumps(value), dtype=object)

    np.savez_compressed(out_path, **payload)
    return out_path


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ckpt = np.load(Path(path), allow_pickle=True)
    state: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    for key in ckpt.files:
        if key == f"{JSON_PREFIX}metadata":
            metadata = json.loads(str(ckpt[key].item()))
            continue

        if key.startswith(JSON_PREFIX):
            raw_key = key[len(JSON_PREFIX) :]
            state[raw_key] = json.loads(str(ckpt[key].item()))
        else:
            state[key] = ckpt[key]

    return state, metadata
