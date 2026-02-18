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


def compute_event_metrics(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    duration_s: float,
    onset_tolerance_s: float,
) -> dict[str, Any]:
    """Compute event precision/recall/F1, FP/hour, and onset delay stats."""
    if onset_tolerance_s < 0:
        raise ValueError(f"onset_tolerance_s must be >= 0, got {onset_tolerance_s}")

    pred_by_seq: dict[str, list[float]] = defaultdict(list)
    gt_by_seq: dict[str, list[float]] = defaultdict(list)

    for pred in predictions:
        sample_id = str(pred.get("sample_id", "global"))
        pred_by_seq[sample_id].append(float(pred["time"]))

    for gt in ground_truth:
        sample_id = str(gt.get("sample_id", "global"))
        gt_by_seq[sample_id].append(float(gt["start"]))

    all_sequences = sorted(set(pred_by_seq) | set(gt_by_seq))

    tp_total = 0
    fp_total = 0
    fn_total = 0
    all_delays: list[float] = []

    for seq in all_sequences:
        tp, fp, fn, delays = _match_events_for_sequence(
            pred_times=pred_by_seq.get(seq, []),
            gt_times=gt_by_seq.get(seq, []),
            onset_tolerance_s=onset_tolerance_s,
        )
        tp_total += tp
        fp_total += fp
        fn_total += fn
        all_delays.extend(delays)

    precision = float(tp_total / max(tp_total + fp_total, 1))
    recall = float(tp_total / max(tp_total + fn_total, 1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-12))

    hours = float(duration_s / 3600.0) if duration_s > 0 else 0.0
    fp_per_hour = float(fp_total / hours) if hours > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp_total),
        "fp": int(fp_total),
        "fn": int(fn_total),
        "fp_per_hour": fp_per_hour,
        "onset_tolerance_s": float(onset_tolerance_s),
        "onset_delay_s": {
            "count": int(len(all_delays)),
            "mean": float(np.mean(all_delays)) if all_delays else 0.0,
            "p50": _safe_percentile(all_delays, 50),
            "p90": _safe_percentile(all_delays, 90),
        },
        "duration_s": float(duration_s),
    }


__all__ = ["compute_event_metrics"]
