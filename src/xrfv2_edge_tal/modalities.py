"""Modality normalization, profile resolution, and channel masking helpers."""

from __future__ import annotations

import re
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np

_CANONICAL_ORDER = ["earbuds", "glasses", "watch", "phone"]
_ALIAS_MAP: dict[str, set[str]] = {
    "earbuds": {
        "earbuds",
        "airpods",
        "earable",
        "bud",
        "buds",
        "pods",
        "imu_earbuds",
    },
    "glasses": {
        "glasses",
        "smart_glasses",
        "smartglasses",
        "imu_glasses",
        "ar_glasses",
        "imu_smart_glasses",
    },
    "watch": {
        "watch",
        "smartwatch",
        "imu_watch",
    },
    "phone": {
        "phone",
        "mobile",
        "imu_phone",
    },
}

_BASE_IMU_FEATURES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


def _normalize_token(name: str) -> str:
    token = re.sub(r"[^a-z0-9_]+", "_", str(name).strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def normalize_modality_name(name: str) -> str:
    """Normalize modality aliases to canonical names when possible."""
    token = _normalize_token(name)
    if token in _ALIAS_MAP:
        return token

    for canonical, aliases in _ALIAS_MAP.items():
        if token in aliases:
            return canonical

    pieces = [piece for piece in token.split("_") if piece]
    for canonical, aliases in _ALIAS_MAP.items():
        if any(piece in aliases for piece in pieces):
            return canonical

    if token.startswith("imu_"):
        stripped = token[4:]
        for canonical, aliases in _ALIAS_MAP.items():
            if stripped == canonical or stripped in aliases:
                return canonical
    return token


def resolve_requested_modalities(
    available_modalities: Iterable[str],
    requested_modalities: Iterable[str] | None,
) -> list[str]:
    """Resolve requested modality names to canonical names and validate availability."""
    available_raw = [str(m) for m in available_modalities]
    available_canonical = OrderedDict.fromkeys(normalize_modality_name(m) for m in available_raw)

    if not requested_modalities:
        resolved = [m for m in _CANONICAL_ORDER if m in available_canonical]
        resolved.extend(m for m in available_canonical if m not in resolved)
        return resolved

    out: list[str] = []
    seen: set[str] = set()
    for req in requested_modalities:
        canonical = normalize_modality_name(str(req))
        if canonical not in available_canonical:
            available_joined = ", ".join(available_raw)
            raise ValueError(
                "Requested modality is not available: "
                f"'{req}' -> '{canonical}'. Available modalities: {available_joined}"
            )
        if canonical not in seen:
            out.append(canonical)
            seen.add(canonical)
    return out


def resolve_modalities_to_raw_keys(
    available_modalities: Iterable[str],
    requested_modalities: Iterable[str] | None,
) -> list[str]:
    """Resolve canonical requested modalities to concrete raw modality keys."""
    raw = [str(m) for m in available_modalities]
    by_canonical: dict[str, list[str]] = {}
    for item in raw:
        by_canonical.setdefault(normalize_modality_name(item), []).append(item)

    wanted = resolve_requested_modalities(raw, requested_modalities)
    out: list[str] = []
    for canonical in wanted:
        out.extend(by_canonical.get(canonical, []))
    return out


def stack_modalities_with_channel_names(
    x_by_modality: dict[str, np.ndarray],
    include_modalities: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a concatenated [T, C] tensor and prefixed channel names."""
    if not x_by_modality:
        raise ValueError("x_by_modality is empty")

    selected_raw = resolve_modalities_to_raw_keys(
        available_modalities=x_by_modality.keys(),
        requested_modalities=include_modalities,
    )
    if not selected_raw:
        raise ValueError("No modalities selected for channel stacking")

    arrays: list[np.ndarray] = []
    channel_names: list[str] = []
    seq_len: int | None = None
    for key in selected_raw:
        arr = np.asarray(x_by_modality[key], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected modality tensor [T, D] for '{key}', got shape {arr.shape}")
        if seq_len is None:
            seq_len = int(arr.shape[0])
        elif int(arr.shape[0]) != seq_len:
            raise ValueError("All modality tensors must share the same temporal length")

        canonical = normalize_modality_name(key)
        feat_names = [
            _BASE_IMU_FEATURES[idx] if idx < len(_BASE_IMU_FEATURES) else f"f{idx}"
            for idx in range(int(arr.shape[1]))
        ]
        channel_names.extend(f"{canonical}:{feat_name}" for feat_name in feat_names)
        arrays.append(arr)

    return np.concatenate(arrays, axis=1), channel_names


def mask_channels_by_profile(
    x: np.ndarray,
    channel_names: list[str],
    include_modalities: Iterable[str],
) -> tuple[np.ndarray, list[str]]:
    """Mask concatenated channels to include only selected modalities."""
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected x with shape [T, C], got {arr.shape}")
    if arr.shape[1] != len(channel_names):
        raise ValueError(
            f"channel_names length ({len(channel_names)}) must match x channels ({arr.shape[1]})"
        )

    allowed = set(resolve_requested_modalities(
        available_modalities=[name.split(":", 1)[0] for name in channel_names],
        requested_modalities=include_modalities,
    ))

    keep_idx: list[int] = []
    new_names: list[str] = []
    for idx, name in enumerate(channel_names):
        prefix = name.split(":", 1)[0]
        canonical = normalize_modality_name(prefix)
        if canonical in allowed:
            keep_idx.append(idx)
            new_names.append(name)

    if not keep_idx:
        available = sorted({normalize_modality_name(name.split(":", 1)[0]) for name in channel_names})
        raise ValueError(
            "No channels matched requested profile modalities "
            f"{sorted(allowed)}. Available modalities from channels: {available}"
        )

    return arr[:, keep_idx], new_names


__all__ = [
    "mask_channels_by_profile",
    "normalize_modality_name",
    "resolve_modalities_to_raw_keys",
    "resolve_requested_modalities",
    "stack_modalities_with_channel_names",
]
