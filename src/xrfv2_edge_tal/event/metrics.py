"""Product-style event detection metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def _match_events_for_sequence(
    pred_times: list[float],
    gt_times: list[float],
    onset_tolerance_s: float,
) -> tuple[int, int, int, list[float]]:
    unmatched_gt = set(range(len(gt_times)))
    tp = 0
    fp = 0
    delays: list[float] = []

    for pred_time in sorted(pred_times):
        best_idx: int | None = None
        best_abs = float("inf")
        for gt_idx in unmatched_gt:
            diff = float(pred_time - gt_times[gt_idx])
            abs_diff = abs(diff)
            if abs_diff <= onset_tolerance_s and abs_diff < best_abs:
                best_abs = abs_diff
                best_idx = gt_idx
        if best_idx is None:
            fp += 1
            continue

        unmatched_gt.remove(best_idx)
        tp += 1
        delays.append(float(pred_time - gt_times[best_idx]))

    fn = len(unmatched_gt)
    return tp, fp, fn, delays


def _match_within_segments_for_sequence(
    pred_times: list[float],
    gt_segments: list[tuple[float, float]],
) -> tuple[int, int, int, list[float]]:
    unmatched_gt = set(range(len(gt_segments)))
    tp = 0
    fp = 0
    delays: list[float] = []

    for pred_time in sorted(pred_times):
        matched_idx = None
        for gt_idx in sorted(unmatched_gt):
            start, end = gt_segments[gt_idx]
            if start <= pred_time <= end:
                matched_idx = gt_idx
                break
        if matched_idx is None:
            fp += 1
            continue

        unmatched_gt.remove(matched_idx)
        tp += 1
        delays.append(float(pred_time - gt_segments[matched_idx][0]))

    fn = len(unmatched_gt)
    return tp, fp, fn, delays


def _mode_payload(
    tp: int,
    fp: int,
    fn: int,
    delays: list[float],
    duration_s: float,
    onset_tolerance_s: float | None = None,
) -> dict[str, Any]:
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))
    hours = float(duration_s / 3600.0) if duration_s > 0 else 0.0
    fp_per_hour = float(fp / hours) if hours > 0 else 0.0
    out = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "fp_per_hour": fp_per_hour,
        "onset_delay_s": {
            "count": int(len(delays)),
            "mean": float(np.mean(delays)) if delays else 0.0,
            "p50": _safe_percentile(delays, 50),
            "p90": _safe_percentile(delays, 90),
        },
    }
    if onset_tolerance_s is not None:
        out["onset_tolerance_s"] = float(onset_tolerance_s)
    return out


def compute_event_metrics(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    duration_s: float,
    onset_tolerance_s: float,
) -> dict[str, Any]:
    """Compute event metrics for onset-strict and within-segment matching."""
    if onset_tolerance_s < 0:
        raise ValueError(f"onset_tolerance_s must be >= 0, got {onset_tolerance_s}")

    pred_by_seq: dict[str, list[float]] = defaultdict(list)
    gt_by_seq: dict[str, list[float]] = defaultdict(list)
    gt_segments_by_seq: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for pred in predictions:
        sample_id = str(pred.get("sample_id", "global"))
        pred_by_seq[sample_id].append(float(pred["time"]))

    for gt in ground_truth:
        sample_id = str(gt.get("sample_id", "global"))
        start = float(gt["start"])
        end = float(gt.get("end", start))
        if end < start:
            end = start
        gt_by_seq[sample_id].append(start)
        gt_segments_by_seq[sample_id].append((start, end))

    all_sequences = sorted(set(pred_by_seq) | set(gt_by_seq))

    onset_tp = 0
    onset_fp = 0
    onset_fn = 0
    onset_delays: list[float] = []

    within_tp = 0
    within_fp = 0
    within_fn = 0
    within_delays: list[float] = []

    for seq in all_sequences:
        tp, fp, fn, delays = _match_events_for_sequence(
            pred_times=pred_by_seq.get(seq, []),
            gt_times=gt_by_seq.get(seq, []),
            onset_tolerance_s=onset_tolerance_s,
        )
        onset_tp += tp
        onset_fp += fp
        onset_fn += fn
        onset_delays.extend(delays)

        tp_w, fp_w, fn_w, delays_w = _match_within_segments_for_sequence(
            pred_times=pred_by_seq.get(seq, []),
            gt_segments=gt_segments_by_seq.get(seq, []),
        )
        within_tp += tp_w
        within_fp += fp_w
        within_fn += fn_w
        within_delays.extend(delays_w)

    return {
        "duration_s": float(duration_s),
        "onset_strict": _mode_payload(
            tp=onset_tp,
            fp=onset_fp,
            fn=onset_fn,
            delays=onset_delays,
            duration_s=duration_s,
            onset_tolerance_s=onset_tolerance_s,
        ),
        "within_segment": _mode_payload(
            tp=within_tp,
            fp=within_fp,
            fn=within_fn,
            delays=within_delays,
            duration_s=duration_s,
            onset_tolerance_s=None,
        ),
    }


__all__ = ["compute_event_metrics"]
