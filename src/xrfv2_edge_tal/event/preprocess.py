"""Preprocessing helpers for event pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_modalities(
    x_dict: dict[str, np.ndarray],
    enabled: bool = True,
    eps: float = 1e-6,
    clip: float | None = None,
) -> dict[str, np.ndarray]:
    """Per-sample, per-modality feature normalization over time."""
    if not enabled:
        return {k: np.asarray(v, dtype=np.float32) for k, v in x_dict.items()}

    out: dict[str, np.ndarray] = {}
    for key, value in x_dict.items():
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected modality tensor [T, D] for '{key}', got {arr.shape}")
        mean = np.mean(arr, axis=0, keepdims=True)
        std = np.std(arr, axis=0, keepdims=True)
        z = (arr - mean) / np.maximum(std, eps)
        if clip is not None and clip > 0:
            z = np.clip(z, -float(clip), float(clip))
        out[key] = z.astype(np.float32, copy=False)
    return out


def normalization_config(data_cfg: dict[str, Any]) -> tuple[bool, float | None]:
    enabled = bool(data_cfg.get("normalize_per_sample", True))
    clip_raw = data_cfg.get("normalize_clip", None)
    clip = float(clip_raw) if clip_raw is not None else None
    return enabled, clip


__all__ = ["normalize_modalities", "normalization_config"]
