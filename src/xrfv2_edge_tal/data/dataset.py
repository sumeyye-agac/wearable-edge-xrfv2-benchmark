"""Dataset wrappers and batch collation."""

from __future__ import annotations

from typing import Any

import numpy as np

from xrfv2_edge_tal.data.adapters import RawAdapter, Segment


class TALDataset:
    """Thin dataset wrapper for adapter + split."""

    def __init__(
        self, adapter: RawAdapter, split: str, sample_ids: list[str] | None = None
    ) -> None:
        self.adapter = adapter
        self.split = split
        self.sample_ids = sample_ids if sample_ids is not None else adapter.split_ids(split)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_id = self.sample_ids[idx]
        x, segments, meta = self.adapter.get_sample(sample_id, self.split)
        return {
            "sample_id": sample_id,
            "x": x,
            "segments": segments,
            "meta": meta,
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length sequence items into a simple dict-of-lists batch."""
    if not items:
        return {"sample_ids": [], "x": {}, "segments": [], "meta": []}

    modalities = list(items[0]["x"].keys())
    batched_x: dict[str, list[np.ndarray]] = {m: [] for m in modalities}
    for item in items:
        for modality in modalities:
            batched_x[modality].append(item["x"][modality])

    return {
        "sample_ids": [item["sample_id"] for item in items],
        "x": batched_x,
        "segments": [item["segments"] for item in items],
        "meta": [item["meta"] for item in items],
    }


__all__ = ["TALDataset", "collate_batch", "Segment"]
