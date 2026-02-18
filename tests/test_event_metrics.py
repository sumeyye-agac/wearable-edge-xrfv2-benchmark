from __future__ import annotations

import numpy as np

from xrfv2_edge_tal.event.metrics import compute_event_metrics
from xrfv2_edge_tal.event.trigger import frame_probs_to_event_triggers


def test_trigger_cooldown_and_hysteresis() -> None:
    probs = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.85, 0.9, 0.1], dtype=np.float32)
    triggers = frame_probs_to_event_triggers(
        probs=probs,
        frame_time_s=0.1,
        threshold=0.7,
        smoothing_window=1,
        cooldown_s=0.25,
        hysteresis=True,
        threshold_off=0.45,
    )
    assert len(triggers) == 1
    assert triggers[0]["frame"] == 2


def test_event_metrics_tp_fp_fn_and_delays() -> None:
    preds = [
        {"sample_id": "0", "time": 1.05},
        {"sample_id": "0", "time": 3.20},
        {"sample_id": "1", "time": 2.40},
    ]
    gts = [
        {"sample_id": "0", "start": 1.00},
        {"sample_id": "0", "start": 2.90},
        {"sample_id": "1", "start": 2.00},
        {"sample_id": "1", "start": 4.00},
    ]

    metrics = compute_event_metrics(
        predictions=preds,
        ground_truth=gts,
        duration_s=3600.0,
        onset_tolerance_s=0.35,
    )

    # Matches: 1.05->1.00 and 3.20->2.90. Third pred is FP, one GT left as FN.
    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["fn"] == 2
    assert abs(metrics["precision"] - (2 / 3)) < 1e-6
    assert abs(metrics["recall"] - 0.5) < 1e-6
    assert abs(metrics["fp_per_hour"] - 1.0) < 1e-6
    assert metrics["onset_delay_s"]["count"] == 2
    assert metrics["onset_delay_s"]["p90"] >= metrics["onset_delay_s"]["p50"]
