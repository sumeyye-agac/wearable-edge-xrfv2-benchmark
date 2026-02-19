"""Candidate generation utilities for hierarchical event detection."""

from __future__ import annotations

from typing import Any

import numpy as np


def motion_energy_from_glasses(x_gl: np.ndarray) -> np.ndarray:
    """Compute per-frame motion energy from glasses IMU."""
    arr = np.asarray(x_gl, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected glasses tensor [T, D], got {arr.shape}")
    if arr.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)

    if arr.shape[1] >= 6:
        gyro = arr[:, 3:6]
        energy = np.sqrt(np.sum(gyro * gyro, axis=1))
    else:
        energy = np.sqrt(np.sum(arr * arr, axis=1))
    return energy.astype(np.float32, copy=False)


def detect_candidates(
    energy: np.ndarray,
    thr: float,
    min_active_s: float,
    cooldown_s: float,
    frame_time_s: float,
) -> list[tuple[int, int]]:
    """Detect contiguous active windows from motion energy."""
    if frame_time_s <= 0:
        raise ValueError(f"frame_time_s must be > 0, got {frame_time_s}")

    x = np.asarray(energy, dtype=np.float32)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D energy, got {x.shape}")

    active = x >= float(thr)
    min_active_frames = int(np.ceil(max(0.0, float(min_active_s)) / frame_time_s))
    min_active_frames = max(1, min_active_frames)
    cooldown_frames = int(np.ceil(max(0.0, float(cooldown_s)) / frame_time_s))

    runs: list[tuple[int, int]] = []
    start = -1
    for idx, flag in enumerate(active):
        if flag and start < 0:
            start = idx
        elif (not flag) and start >= 0:
            if idx - start >= min_active_frames:
                runs.append((start, idx))
            start = -1
    if start >= 0 and active.shape[0] - start >= min_active_frames:
        runs.append((start, int(active.shape[0])))

    if cooldown_frames <= 0:
        return runs

    merged: list[tuple[int, int]] = []
    last_end = -(10**9)
    for run_start, run_end in runs:
        if run_start - last_end < cooldown_frames:
            if merged:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, max(prev_end, run_end))
                last_end = merged[-1][1]
            continue
        merged.append((run_start, run_end))
        last_end = run_end
    return merged


def widen_window(
    start: int,
    end: int,
    pre_s: float,
    post_s: float,
    t_frames: int,
    frame_time_s: float,
) -> tuple[int, int]:
    """Expand a candidate window with pre/post context while respecting boundaries."""
    if frame_time_s <= 0:
        raise ValueError(f"frame_time_s must be > 0, got {frame_time_s}")
    if t_frames <= 0:
        return 0, 0
    if end <= start:
        return max(0, min(start, t_frames)), max(0, min(start + 1, t_frames))

    pre_frames = int(np.round(max(0.0, float(pre_s)) / frame_time_s))
    post_frames = int(np.round(max(0.0, float(post_s)) / frame_time_s))
    out_start = max(0, int(start) - pre_frames)
    out_end = min(int(t_frames), int(end) + post_frames)
    if out_end <= out_start:
        out_end = min(int(t_frames), out_start + 1)
    return out_start, out_end


def default_candidate_config(raw_cfg: dict[str, Any] | None = None) -> dict[str, float | int]:
    cfg = raw_cfg or {}
    return {
        "energy_threshold": float(cfg.get("energy_threshold", 1.0)),
        "min_active_s": float(cfg.get("min_active_s", 0.2)),
        "cooldown_s": float(cfg.get("cooldown_s", 0.5)),
        "pre_s": float(cfg.get("pre_s", 0.2)),
        "post_s": float(cfg.get("post_s", 0.4)),
        "window_len_s": float(cfg.get("window_len_s", 1.5)),
        "overlap_min_s": float(cfg.get("overlap_min_s", 0.2)),
        "max_windows": int(cfg.get("max_windows", 24)),
        "include_gt_windows": bool(cfg.get("include_gt_windows", True)),
    }


__all__ = [
    "default_candidate_config",
    "detect_candidates",
    "motion_energy_from_glasses",
    "widen_window",
]
