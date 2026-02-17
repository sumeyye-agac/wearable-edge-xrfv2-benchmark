"""Frame-wise to segment decoding utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

SegmentLike = dict[str, Any]


def _smooth_probs(probs: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return probs
    kernel = int(max(1, kernel))
    pad = kernel // 2
    padded = np.pad(probs, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(probs)
    for t in range(probs.shape[0]):
        out[t] = np.mean(padded[t : t + kernel], axis=0)
    return out


def _merge_segments(segments: list[SegmentLike], min_gap: int) -> list[SegmentLike]:
    if min_gap <= 0 or not segments:
        return segments
    segments = sorted(segments, key=lambda s: (int(s["label"]), float(s["start"])))
    merged: list[SegmentLike] = [dict(segments[0])]
    for seg in segments[1:]:
        last = merged[-1]
        if int(seg["label"]) == int(last["label"]) and float(seg["start"]) - float(last["end"]) <= min_gap:
            new_end = float(seg["end"])
            old_len = max(1e-6, float(last["end"]) - float(last["start"]))
            new_len = max(1e-6, new_end - float(seg["start"]))
            last["score"] = float(
                (float(last["score"]) * old_len + float(seg["score"]) * new_len) / max(old_len + new_len, 1e-6)
            )
            last["end"] = new_end
        else:
            merged.append(dict(seg))
    return merged


def decode_per_class_threshold(
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


def decode_argmax_probs(
    probs: np.ndarray,
    score_threshold: float = 0.5,
    min_len: int = 3,
    background_class: int = 0,
    smooth_kernel: int = 1,
    min_gap: int = 0,
) -> list[SegmentLike]:
    """Decode one-label-per-frame segments with confidence thresholding."""
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape [T, C], got {probs.shape}")

    smoothed = _smooth_probs(probs, kernel=smooth_kernel)
    conf = np.max(smoothed, axis=1)
    labels = np.argmax(smoothed, axis=1).astype(np.int64)
    if background_class >= 0:
        labels = np.where(conf >= score_threshold, labels, background_class)
    else:
        labels = np.where(conf >= score_threshold, labels, -1)

    t_len = int(smoothed.shape[0])
    out: list[SegmentLike] = []
    start = 0
    for t in range(1, t_len + 1):
        boundary = t == t_len or labels[t] != labels[start]
        if not boundary:
            continue
        label = int(labels[start])
        end = t
        if label >= 0 and label != background_class and (end - start) >= min_len:
            seg_score = float(np.mean(smoothed[start:end, label]))
            out.append(
                {
                    "label": label,
                    "start": float(start),
                    "end": float(end),
                    "score": seg_score,
                }
            )
        start = t

    return _merge_segments(out, min_gap=min_gap)


def decode_framewise_probs(
    probs: np.ndarray,
    score_threshold: float = 0.5,
    min_len: int = 3,
    background_class: int = 0,
) -> list[SegmentLike]:
    """Back-compat alias for per-class threshold decoding."""
    return decode_per_class_threshold(
        probs=probs,
        score_threshold=score_threshold,
        min_len=min_len,
        background_class=background_class,
    )
