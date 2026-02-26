from __future__ import annotations

import numpy as np

from xrfv2_edge_tal.event.train_event import _rebalance_windows


def _make_windows(num_pos: int, num_neg: int) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    out.extend({"y": 1, "id": idx} for idx in range(num_pos))
    out.extend({"y": 0, "id": num_pos + idx} for idx in range(num_neg))
    return out


def test_rebalance_windows_ratio_applies_cap() -> None:
    rng = np.random.default_rng(42)
    windows = _make_windows(num_pos=2, num_neg=10)
    merged, pos_count, neg_before, neg_after = _rebalance_windows(
        windows,
        max_negative_ratio=2.0,
        rng=rng,
    )
    assert pos_count == 2
    assert neg_before == 10
    assert neg_after == 4
    assert len(merged) == 6
    assert sum(int(w["y"]) for w in merged) == 2


def test_rebalance_windows_disabled_keeps_all() -> None:
    rng = np.random.default_rng(7)
    windows = _make_windows(num_pos=3, num_neg=5)
    merged, pos_count, neg_before, neg_after = _rebalance_windows(
        windows,
        max_negative_ratio=0.0,
        rng=rng,
    )
    assert pos_count == 3
    assert neg_before == 5
    assert neg_after == 5
    assert len(merged) == 8


def test_rebalance_windows_no_positive_leaves_negatives() -> None:
    rng = np.random.default_rng(3)
    windows = _make_windows(num_pos=0, num_neg=6)
    merged, pos_count, neg_before, neg_after = _rebalance_windows(
        windows,
        max_negative_ratio=3.0,
        rng=rng,
    )
    assert pos_count == 0
    assert neg_before == 6
    assert neg_after == 6
    assert len(merged) == 6
