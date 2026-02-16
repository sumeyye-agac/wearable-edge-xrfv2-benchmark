from __future__ import annotations

from xrfv2_edge_tal.metrics.tal_map import ap_at_tiou, map_over_thresholds, segment_tiou
from xrfv2_edge_tal.postprocess.nms import temporal_nms


def test_segment_tiou_known_values() -> None:
    a = {"start": 0.0, "end": 10.0}
    b = {"start": 5.0, "end": 15.0}
    # overlap=5, union=15 => 1/3
    assert abs(segment_tiou(a, b) - (1.0 / 3.0)) < 1e-6


def test_ap_behavior() -> None:
    gts = [
        {"sample_id": "0", "label": 1, "start": 0.0, "end": 10.0},
        {"sample_id": "0", "label": 2, "start": 20.0, "end": 30.0},
    ]
    preds = [
        {"sample_id": "0", "label": 1, "start": 0.0, "end": 10.0, "score": 0.95},
        {"sample_id": "0", "label": 1, "start": 1.0, "end": 9.0, "score": 0.70},
        {"sample_id": "0", "label": 2, "start": 20.0, "end": 30.0, "score": 0.90},
        {"sample_id": "0", "label": 2, "start": 0.0, "end": 8.0, "score": 0.10},
    ]

    ap = ap_at_tiou(preds, gts, tiou=0.5)
    assert 0.95 <= ap <= 1.0


def test_map_over_thresholds_and_nms() -> None:
    gts = [{"sample_id": "0", "label": 1, "start": 0.0, "end": 10.0}]
    preds = [
        {"sample_id": "0", "label": 1, "start": 0.0, "end": 10.0, "score": 0.9},
        {"sample_id": "0", "label": 1, "start": 1.0, "end": 9.0, "score": 0.8},
    ]

    filtered = temporal_nms(preds, tiou_threshold=0.5, classwise=True)
    assert len(filtered) == 1

    m = map_over_thresholds(filtered, gts)
    assert "map_avg" in m
    assert 0.0 <= m["map_avg"] <= 1.0
