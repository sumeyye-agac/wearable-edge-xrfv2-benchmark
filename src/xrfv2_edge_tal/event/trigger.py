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
    min_active_s: float = 0.0,
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
    min_active_frames = int(np.ceil(max(0.0, float(min_active_s)) / frame_time_s))
    if min_active_frames > 1:
        filtered = np.zeros_like(active, dtype=bool)
        run_start = -1
        for idx, state in enumerate(active):
            if state and run_start < 0:
                run_start = idx
            elif (not state) and run_start >= 0:
                if idx - run_start >= min_active_frames:
                    filtered[run_start:idx] = True
                run_start = -1
        if run_start >= 0 and active.shape[0] - run_start >= min_active_frames:
            filtered[run_start:] = True
        active = filtered

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


def filter_trigger_candidates(
    candidates: list[dict[str, float | int]],
    threshold: float,
    cooldown_s: float,
    hysteresis: bool = False,
    threshold_off: float | None = None,
) -> list[dict[str, float | int]]:
    """Filter candidate trigger points by score threshold and cooldown."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda x: float(x["time"]))
    off = float(threshold_off if threshold_off is not None else threshold * 0.7)
    if off > threshold:
        raise ValueError("threshold_off must be <= threshold when hysteresis is enabled")

    out: list[dict[str, float | int]] = []
    last_time = -1e18
    active = False
    for item in ordered:
        score = float(item["score"])
        ts = float(item["time"])
        if hysteresis:
            if active and score <= off:
                active = False
            if (not active) and score >= threshold:
                if ts - last_time >= max(0.0, cooldown_s):
                    out.append(item)
                    last_time = ts
                    active = True
        else:
            if score < threshold:
                continue
            if ts - last_time < max(0.0, cooldown_s):
                continue
            out.append(item)
            last_time = ts
    return out


__all__ = [
    "filter_trigger_candidates",
    "frame_probs_to_event_triggers",
    "smooth_1d",
    "threshold_with_hysteresis",
]
