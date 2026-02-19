"""XRF V2 dataset shape probing utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from xrfv2_edge_tal.data.adapters import XRFV2H5Adapter


def _collect_h5_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    out: list[str] = []
    for key, obj in group.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(obj, h5py.Dataset):
            out.append(path)
        elif isinstance(obj, h5py.Group):
            out.extend(_collect_h5_paths(obj, path))
    return out


def _resolve_dataset(h5f: h5py.File, key: str) -> h5py.Dataset | None:
    if key in h5f and isinstance(h5f[key], h5py.Dataset):
        return h5f[key]
    for path in _collect_h5_paths(h5f):
        if path == key or path.endswith(f"/{key}"):
            obj = h5f[path]
            if isinstance(obj, h5py.Dataset):
                return obj
    return None


def _sample_stats(arr: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(arr, dtype=np.float32)
    return {
        "shape": list(finite.shape),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "has_nan": bool(np.isnan(finite).any()),
    }


def probe_xrfv2_h5(data_root: str | Path, sample_index: int = 0) -> dict[str, Any]:
    """Probe dataset keys, shapes, and sample stats from XRF V2 H5 files."""
    root = Path(data_root)
    adapter = XRFV2H5Adapter(root)

    train_h5 = root / "train_data.h5"
    with h5py.File(train_h5, "r") as h5f:
        paths = sorted(_collect_h5_paths(h5f))
        dataset_shapes: dict[str, list[int]] = {}
        for key in ["imu", "wifi", "airpods"]:
            ds = _resolve_dataset(h5f, key)
            if ds is not None:
                dataset_shapes[key] = list(ds.shape)

    split_ids = adapter.split_ids("train")
    if not split_ids:
        raise ValueError(f"No train samples found in {root}")
    if sample_index < 0 or sample_index >= len(split_ids):
        raise IndexError(f"sample_index {sample_index} out of range [0, {len(split_ids) - 1}]")

    sample_id = split_ids[sample_index]
    x, _, _ = adapter.get_sample(sample_id, "train")
    sample_shapes = {key: list(np.asarray(value).shape) for key, value in x.items()}
    sample_stats = {key: _sample_stats(np.asarray(value)) for key, value in x.items()}

    return {
        "data_root": str(root),
        "train_h5": str(train_h5),
        "h5_paths": paths,
        "dataset_shapes": dataset_shapes,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "sample_shapes": sample_shapes,
        "sample_stats": sample_stats,
        "adapter_modalities": adapter.modalities,
        "counts": {
            "train": len(adapter.split_ids("train")),
            "test": len(adapter.split_ids("test")),
        },
    }


def probe_xrfv2_h5_json(data_root: str | Path, sample_index: int = 0) -> str:
    payload = probe_xrfv2_h5(data_root=data_root, sample_index=sample_index)
    return json.dumps(payload, indent=2, sort_keys=True)


__all__ = ["probe_xrfv2_h5", "probe_xrfv2_h5_json"]
