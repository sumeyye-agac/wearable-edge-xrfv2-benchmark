from __future__ import annotations

import numpy as np
import pytest

from xrfv2_edge_tal.event.eval_event import (
    _candidate_frame,
    _candidate_score,
    _normalize_candidate_scores,
    _resolve_hierarchical_cfg,
)


def test_candidate_score_modes() -> None:
    probs = np.asarray([0.1, 0.2, 0.9, 0.4], dtype=np.float32)
    assert _candidate_score(probs, "mean") == pytest.approx(0.4)
    assert _candidate_score(probs, "max") == pytest.approx(0.9)
    assert _candidate_score(probs, "p90") > 0.7
    assert _candidate_score(probs, "p95") > 0.8


def test_candidate_frame_modes() -> None:
    probs = np.asarray([0.1, 0.2, 0.9, 0.4], dtype=np.float32)
    assert _candidate_frame(start_frame=10, pos_probs=probs, trigger_time="start") == 10
    assert _candidate_frame(start_frame=10, pos_probs=probs, trigger_time="peak") == 12


def test_resolve_hierarchical_cfg_validates_new_fields() -> None:
    cfg = {
        "eval": {
            "hierarchical": {
                "score_mode": "max",
                "trigger_time": "peak",
                "score_normalization": "center_median",
            }
        }
    }
    out = _resolve_hierarchical_cfg(cfg)
    assert out["score_mode"] == "max"
    assert out["trigger_time"] == "peak"
    assert out["score_normalization"] == "center_median"

    bad_score = {"eval": {"hierarchical": {"score_mode": "bad"}}}
    with pytest.raises(ValueError, match="score_mode"):
        _resolve_hierarchical_cfg(bad_score)

    bad_time = {"eval": {"hierarchical": {"trigger_time": "bad"}}}
    with pytest.raises(ValueError, match="trigger_time"):
        _resolve_hierarchical_cfg(bad_time)

    bad_norm = {"eval": {"hierarchical": {"score_normalization": "bad"}}}
    with pytest.raises(ValueError, match="score_normalization"):
        _resolve_hierarchical_cfg(bad_norm)


def test_normalize_candidate_scores_center_median() -> None:
    scored = [
        {"time": 0.1, "score": 0.4, "frame": 1},
        {"time": 0.2, "score": 0.6, "frame": 2},
        {"time": 0.3, "score": 0.7, "frame": 3},
    ]
    out = _normalize_candidate_scores(scored, "center_median")
    values = np.asarray([float(x["score"]) for x in out], dtype=np.float32)
    assert np.median(values) == pytest.approx(0.0, abs=1e-6)
