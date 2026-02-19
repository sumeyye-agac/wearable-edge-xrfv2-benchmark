"""Event trigger generation from framewise probabilities."""

from __future__ import annotations

from typing import Any

import numpy as np


def smooth_1d(signal: np.ndarray, window: int) -> np.ndarray:
    """Simple moving-average smoothing for 1D probabilities."""
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {x.shape}")
    if window <= 1:
        return x
    w = int(max(1, window))
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    out = np.zeros_like(x)
    for idx in range(x.shape[0]):
        out[idx] = np.mean(padded[idx : idx + w])
    return out


def threshold_with_hysteresis(
    probs: np.ndarray,
    threshold_on: float,
    threshold_off: float,
) -> np.ndarray:
    """Stateful hysteresis thresholding."""
    x = np.asarray(probs, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D probabilities, got shape {x.shape}")
    active = np.zeros((x.shape[0],), dtype=bool)
    on = False
    for idx, value in enumerate(x):
        if not on and value >= threshold_on:
            on = True
        elif on and value <= threshold_off:
            on = False
        active[idx] = on
    return active


def frame_probs_to_event_triggers(
    probs: np.ndarray,
    frame_time_s: float,
    threshold: float = 0.5,
    smoothing_window: int = 5,
    cooldown_s: float = 1.0,
    hysteresis: bool = False,
    threshold_off: float | None = None,
) -> list[dict[str, Any]]:
    """Convert framewise probabilities into discrete event triggers."""
    if frame_time_s <= 0:
        raise ValueError(f"frame_time_s must be > 0, got {frame_time_s}")

    x = np.asarray(probs, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D probabilities, got shape {x.shape}")

    smoothed = smooth_1d(x, window=smoothing_window)
    off = float(threshold_off if threshold_off is not None else threshold * 0.7)
    if off > threshold:
        raise ValueError("threshold_off must be <= threshold when hysteresis is enabled")

    if hysteresis:
        active = threshold_with_hysteresis(smoothed, threshold_on=threshold, threshold_off=off)
    else:
        active = smoothed >= threshold

    rising_edges: list[int] = []
    prev = False
    for idx, state in enumerate(active):
        if state and not prev:
            rising_edges.append(idx)
        prev = bool(state)

    cooldown_frames = int(np.ceil(cooldown_s / frame_time_s))
    triggers: list[dict[str, Any]] = []
    last_kept_frame = -(10**9)
    for frame_idx in rising_edges:
        if frame_idx - last_kept_frame <= cooldown_frames:
            continue
        triggers.append(
            {
                "frame": int(frame_idx),
                "time": float(frame_idx * frame_time_s),
                "score": float(smoothed[frame_idx]),
            }
        )
        last_kept_frame = frame_idx
    return triggers


__all__ = [
    "frame_probs_to_event_triggers",
    "smooth_1d",
    "threshold_with_hysteresis",
]
