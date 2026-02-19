"""Window construction for hierarchical event training."""

from __future__ import annotations

from typing import Any

import numpy as np

from xrfv2_edge_tal.event.candidates import (
    default_candidate_config,
    detect_candidates,
    motion_energy_from_glasses,
    widen_window,
)


def _window_frames(window_len_s: float, frame_time_s: float) -> int:
    if frame_time_s <= 0:
        raise ValueError(f"frame_time_s must be > 0, got {frame_time_s}")
    raw = int(np.round(max(0.1, float(window_len_s)) / frame_time_s))
    return int(np.clip(raw, 16, 512))


def _segment_to_frames(seg: dict[str, float | int], seq_len: int) -> tuple[int, int]:
    start = float(seg["start"])
    end = float(seg["end"])
    if start <= 1.0 and end <= 1.0:
        start *= seq_len
        end *= seq_len
    s = int(np.floor(start))
    e = int(np.ceil(end))
    s = max(0, min(s, seq_len))
    e = max(s + 1, min(e, seq_len))
    return s, e


def _slice_or_pad(arr: np.ndarray, start: int, end: int, target_len: int) -> np.ndarray:
    clip_start = max(0, int(start))
    clip_end = min(int(arr.shape[0]), int(end))
    if clip_end <= clip_start:
        clip_end = min(int(arr.shape[0]), clip_start + 1)
    part = arr[clip_start:clip_end]
    if part.shape[0] == target_len:
        return part.astype(np.float32, copy=False)
    if part.shape[0] > target_len:
        return part[:target_len].astype(np.float32, copy=False)
    pad_rows = target_len - part.shape[0]
    if part.shape[0] == 0:
        base = np.zeros((1, arr.shape[1]), dtype=np.float32)
    else:
        base = part[-1:, :].astype(np.float32, copy=False)
    pad = np.repeat(base, pad_rows, axis=0)
    return np.concatenate([part.astype(np.float32, copy=False), pad], axis=0)


def _fixed_window(start: int, end: int, seq_len: int, w_frames: int) -> tuple[int, int]:
    center = int((start + end) // 2)
    out_start = center - (w_frames // 2)
    out_end = out_start + w_frames
    if out_start < 0:
        out_end += -out_start
        out_start = 0
    if out_end > seq_len:
        shift = out_end - seq_len
        out_start = max(0, out_start - shift)
        out_end = seq_len
    if out_end - out_start < w_frames:
        out_start = max(0, out_end - w_frames)
    return int(out_start), int(min(seq_len, out_start + w_frames))


def _window_label(
    *,
    start: int,
    end: int,
    segments: list[dict[str, float | int]],
    positive_label_ids: set[int],
    seq_len: int,
    overlap_min_frames: int,
) -> int:
    for seg in segments:
        if int(seg["label"]) not in positive_label_ids:
            continue
        gs, ge = _segment_to_frames(seg, seq_len=seq_len)
        overlap = max(0, min(end, ge) - max(start, gs))
        if overlap >= overlap_min_frames:
            return 1
    return 0


def build_windows_for_sample(
    x_dict: dict[str, np.ndarray],
    segments: list[dict[str, float | int]],
    positive_ids: set[int],
    profile_modalities: list[str],
    frame_time_s: float,
    candidate_cfg: dict[str, Any] | None,
    window_len_s: float,
    sample_id: str | None = None,
) -> list[dict[str, Any]]:
    cfg = default_candidate_config(candidate_cfg)
    w_frames = _window_frames(window_len_s=window_len_s, frame_time_s=frame_time_s)
    overlap_min_frames = int(
        np.ceil(max(0.0, float(cfg["overlap_min_s"])) / max(frame_time_s, 1e-9))
    )
    overlap_min_frames = max(1, overlap_min_frames)

    if "imu_gl" not in x_dict:
        return []
    seq_len = int(x_dict["imu_gl"].shape[0])
    if seq_len <= 0:
        return []

    energy = motion_energy_from_glasses(x_dict["imu_gl"])
    raw_candidates = detect_candidates(
        energy=energy,
        thr=float(cfg["energy_threshold"]),
        min_active_s=float(cfg["min_active_s"]),
        cooldown_s=float(cfg["cooldown_s"]),
        frame_time_s=frame_time_s,
    )
    expanded: list[tuple[int, int]] = []
    for start, end in raw_candidates:
        s, e = widen_window(
            start=start,
            end=end,
            pre_s=float(cfg["pre_s"]),
            post_s=float(cfg["post_s"]),
            t_frames=seq_len,
            frame_time_s=frame_time_s,
        )
        expanded.append((s, e))

    candidate_windows: list[tuple[int, int]] = []
    seen_starts: set[int] = set()
    for start, end in expanded:
        s, e = _fixed_window(start, end, seq_len=seq_len, w_frames=w_frames)
        if s not in seen_starts:
            candidate_windows.append((s, e))
            seen_starts.add(s)

    gt_windows: list[tuple[int, int]] = []
    if bool(cfg["include_gt_windows"]):
        for seg in segments:
            if int(seg["label"]) not in positive_ids:
                continue
            gs, ge = _segment_to_frames(seg, seq_len=seq_len)
            s, e = _fixed_window(gs, ge, seq_len=seq_len, w_frames=w_frames)
            if s not in seen_starts:
                gt_windows.append((s, e))
                seen_starts.add(s)

    windows = list(gt_windows) + list(candidate_windows)
    max_windows = max(1, int(cfg["max_windows"]))
    windows = windows[:max_windows]

    out: list[dict[str, Any]] = []
    for start, end in windows:
        x_window: dict[str, np.ndarray] = {}
        for modality in profile_modalities:
            if modality not in x_dict:
                continue
            x_window[modality] = _slice_or_pad(
                np.asarray(x_dict[modality], dtype=np.float32),
                start=start,
                end=end,
                target_len=w_frames,
            )
        if not x_window:
            continue
        y = _window_label(
            start=start,
            end=end,
            segments=segments,
            positive_label_ids=positive_ids,
            seq_len=seq_len,
            overlap_min_frames=overlap_min_frames,
        )
        out.append(
            {
                "x_window": x_window,
                "y": int(y),
                "sample_id": sample_id or "",
                "start_frame": int(start),
                "end_frame": int(end),
            }
        )
    return out


__all__ = ["build_windows_for_sample"]
