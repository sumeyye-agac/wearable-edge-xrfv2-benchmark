"""Paper-aligned lightweight training/eval helpers.

This module aligns data handling with the XRFV2 paper protocol while
keeping the model family lightweight and edge-friendly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

Segment = dict[str, float | int]


def to_frame_segments(segments: list[Segment], seq_len: int) -> list[Segment]:
    """Convert normalized [0,1] segments into frame-space if needed."""
    out: list[Segment] = []
    for seg in segments:
        raw_start = float(seg["start"])
        raw_end = float(seg["end"])
        if raw_start <= 1.0 and raw_end <= 1.0:
            raw_start *= float(seq_len)
            raw_end *= float(seq_len)
        out.append(
            {
                "start": float(max(0.0, raw_start)),
                "end": float(min(float(seq_len), raw_end)),
                "label": int(seg["label"]),
            }
        )
    return out


def resample_modality(x: np.ndarray, target_len: int) -> np.ndarray:
    """Resample [T, D] modality sequence with linear interpolation."""
    if target_len <= 0:
        return x
    src_len = int(x.shape[0])
    if src_len == target_len:
        return x
    if src_len <= 1:
        return np.repeat(x, repeats=target_len, axis=0).astype(np.float32, copy=False)

    src_pos = np.linspace(0.0, 1.0, num=src_len, dtype=np.float32)
    dst_pos = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.zeros((target_len, x.shape[1]), dtype=np.float32)
    for col in range(x.shape[1]):
        out[:, col] = np.interp(dst_pos, src_pos, x[:, col].astype(np.float32))
    return out


def resample_sample(
    x_dict: dict[str, np.ndarray],
    segments: list[Segment],
    target_len: int,
) -> tuple[dict[str, np.ndarray], list[Segment]]:
    """Resample all modalities and scale frame-space segment boundaries."""
    if target_len <= 0:
        return x_dict, segments
    first_modality = next(iter(x_dict.keys()))
    src_len = int(x_dict[first_modality].shape[0])
    if src_len == target_len:
        return x_dict, segments

    scale = float(target_len) / float(max(src_len, 1))
    x_out = {k: resample_modality(v, target_len=target_len) for k, v in x_dict.items()}
    seg_out = []
    for seg in segments:
        seg_out.append(
            {
                "start": float(seg["start"]) * scale,
                "end": float(seg["end"]) * scale,
                "label": int(seg["label"]),
            }
        )
    return x_out, seg_out


def window_starts(seq_len: int, clip_len: int, stride: int) -> list[int]:
    """Generate starts that fully cover the timeline."""
    if clip_len <= 0:
        raise ValueError("clip_len must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if seq_len <= clip_len:
        return [0]

    starts = list(range(0, seq_len - clip_len + 1, stride))
    tail_start = seq_len - clip_len
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def _slice_with_edge_pad(
    x: np.ndarray, start: int, clip_len: int, seq_len: int
) -> tuple[np.ndarray, int]:
    end = min(start + clip_len, seq_len)
    valid_len = max(0, end - start)

    out = np.zeros((clip_len, x.shape[1]), dtype=np.float32)
    if valid_len <= 0:
        return out, 0

    out[:valid_len] = x[start:end]
    if valid_len < clip_len:
        out[valid_len:] = out[valid_len - 1]
    return out, valid_len


def _clip_segments_to_window(
    segments: list[Segment],
    start: int,
    clip_len: int,
    min_coverage: float,
) -> list[Segment]:
    end = start + clip_len
    out: list[Segment] = []
    for seg in segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        full = max(seg_end - seg_start, 1e-6)
        inter_start = max(seg_start, float(start))
        inter_end = min(seg_end, float(end))
        inter = max(0.0, inter_end - inter_start)
        if inter <= 0.0:
            continue
        if inter / full < min_coverage:
            continue
        out.append(
            {
                "start": inter_start - float(start),
                "end": inter_end - float(start),
                "label": int(seg["label"]),
            }
        )
    return out


def make_windows(
    x_dict: dict[str, np.ndarray],
    segments: list[Segment],
    clip_len: int,
    stride: int,
    min_coverage: float = 0.25,
) -> list[dict[str, Any]]:
    """Build fixed-length overlapping windows for paper-aligned training/eval."""
    first_modality = next(iter(x_dict.keys()))
    seq_len = int(x_dict[first_modality].shape[0])

    out: list[dict[str, Any]] = []
    for start in window_starts(seq_len=seq_len, clip_len=clip_len, stride=stride):
        window_x: dict[str, np.ndarray] = {}
        valid_len = clip_len
        for modality, x in x_dict.items():
            xw, this_valid = _slice_with_edge_pad(
                x, start=start, clip_len=clip_len, seq_len=seq_len
            )
            window_x[modality] = xw
            valid_len = min(valid_len, this_valid)
        out.append(
            {
                "start": int(start),
                "valid_len": int(valid_len),
                "x": window_x,
                "segments": _clip_segments_to_window(
                    segments=segments,
                    start=start,
                    clip_len=clip_len,
                    min_coverage=min_coverage,
                ),
            }
        )
    return out


def augment_modalities(
    x_dict: dict[str, np.ndarray],
    rng: np.random.Generator,
    noise_std: float = 0.0,
    scale_jitter: float = 0.0,
) -> dict[str, np.ndarray]:
    """Apply light numeric augmentation for robustness."""
    out: dict[str, np.ndarray] = {}
    for modality, x in x_dict.items():
        z = np.array(x, copy=True)
        if scale_jitter > 0.0:
            gain = float(rng.uniform(1.0 - scale_jitter, 1.0 + scale_jitter))
            z *= gain
        if noise_std > 0.0:
            z += rng.normal(0.0, noise_std, size=z.shape).astype(np.float32)
        out[modality] = z.astype(np.float32, copy=False)
    return out


def aggregate_window_probs(
    window_probs: list[np.ndarray],
    starts: list[int],
    valid_lens: list[int],
    full_len: int,
) -> np.ndarray:
    """Average overlapping window probabilities back to full timeline."""
    if not window_probs:
        return np.zeros((full_len, 1), dtype=np.float32)

    num_classes = int(window_probs[0].shape[1])
    agg = np.zeros((full_len, num_classes), dtype=np.float32)
    counts = np.zeros((full_len, 1), dtype=np.float32)

    for probs, start, valid_len in zip(window_probs, starts, valid_lens, strict=True):
        take = max(0, min(int(valid_len), probs.shape[0], full_len - int(start)))
        if take <= 0:
            continue
        agg[start : start + take] += probs[:take]
        counts[start : start + take] += 1.0

    counts = np.maximum(counts, 1.0)
    return agg / counts
