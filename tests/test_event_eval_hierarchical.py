from __future__ import annotations

import numpy as np
import pytest

from xrfv2_edge_tal.event.eval_event import (
    _candidate_frame,
    _candidate_score,
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
    cfg = {"eval": {"hierarchical": {"score_mode": "max", "trigger_time": "peak"}}}
    out = _resolve_hierarchical_cfg(cfg)
    assert out["score_mode"] == "max"
    assert out["trigger_time"] == "peak"

    bad_score = {"eval": {"hierarchical": {"score_mode": "bad"}}}
    with pytest.raises(ValueError, match="score_mode"):
        _resolve_hierarchical_cfg(bad_score)

    bad_time = {"eval": {"hierarchical": {"trigger_time": "bad"}}}
    with pytest.raises(ValueError, match="trigger_time"):
        _resolve_hierarchical_cfg(bad_time)
