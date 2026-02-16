"""Frame-wise to segment decoding utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

SegmentLike = dict[str, Any]


def decode_framewise_probs(
    probs: np.ndarray,
    score_threshold: float = 0.5,
    min_len: int = 3,
    background_class: int = 0,
) -> list[SegmentLike]:
    """Decode framewise class probabilities into contiguous segments.

    Assumes class index 0 is background by default.
    """
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape [T, C], got {probs.shape}")

    t_len, n_classes = probs.shape
    out: list[SegmentLike] = []

    for cls in range(n_classes):
        if cls == background_class:
            continue

        mask = probs[:, cls] >= score_threshold
        start = None
        for t in range(t_len):
            if mask[t] and start is None:
                start = t
            if start is not None and (t == t_len - 1 or not mask[t + 1]):
                end = t + 1
                length = end - start
                if length >= min_len:
                    seg_score = float(np.mean(probs[start:end, cls]))
                    out.append(
                        {
                            "label": int(cls),
                            "start": float(start),
                            "end": float(end),
                            "score": seg_score,
                        }
                    )
                start = None

    return out
