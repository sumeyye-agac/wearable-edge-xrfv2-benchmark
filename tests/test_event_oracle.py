from __future__ import annotations

from typing import Any

import numpy as np

from xrfv2_edge_tal.event.metrics import compute_event_metrics
from xrfv2_edge_tal.event.trigger import frame_probs_to_event_triggers


def _within_segment_match(
    predictions: list[dict[str, Any]],
    gt_segments: list[dict[str, Any]],
) -> dict[str, float]:
    remaining = set(range(len(gt_segments)))
    tp = 0
    fp = 0

    for pred in sorted(predictions, key=lambda row: float(row["time"])):
        matched_idx = None
        for idx in remaining:
            gt = gt_segments[idx]
            if float(gt["start"]) <= float(pred["time"]) <= float(gt["end"]):
                matched_idx = idx
                break
        if matched_idx is None:
            fp += 1
            continue
        remaining.remove(matched_idx)
        tp += 1

    fn = len(remaining)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def test_oracle_perfect_probs_yield_high_event_metrics() -> None:
    frame_time_s = 0.02
    seq_len = 120
    gt_frames = [(10, 20), (50, 70), (90, 100)]

    probs = np.zeros((seq_len,), dtype=np.float32)
    for start, end in gt_frames:
        probs[start:end] = 1.0

    triggers = frame_probs_to_event_triggers(
        probs=probs,
        frame_time_s=frame_time_s,
        threshold=0.5,
        smoothing_window=1,
        cooldown_s=0.1,
        hysteresis=False,
    )

    predictions = [
        {"sample_id": "0", "time": float(item["time"]), "score": float(item["score"])}
        for item in triggers
    ]
    gt_events = [{"sample_id": "0", "start": float(start * frame_time_s)} for start, _ in gt_frames]
    gt_segments = [
        {"sample_id": "0", "start": float(start * frame_time_s), "end": float(end * frame_time_s)}
        for start, end in gt_frames
    ]

    metrics = compute_event_metrics(
        predictions=predictions,
        ground_truth=gt_events,
        duration_s=float(seq_len * frame_time_s),
        onset_tolerance_s=0.04,
    )
    onset = metrics["onset_strict"]
    within_mode = metrics["within_segment"]
    within = _within_segment_match(predictions=predictions, gt_segments=gt_segments)

    assert onset["precision"] >= 0.99
    assert onset["recall"] >= 0.99
    assert onset["f1"] >= 0.99
    assert within_mode["f1"] >= 0.99
    assert within["f1"] >= 0.99
