from __future__ import annotations

import numpy as np

from xrfv2_edge_tal.decoding import decode_argmax_probs, decode_per_class_threshold


def test_decode_argmax_basic() -> None:
    probs = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.1, 0.9, 0.0],
            [0.2, 0.7, 0.1],
            [0.7, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    segs = decode_argmax_probs(
        probs=probs,
        score_threshold=0.5,
        min_len=2,
        background_class=0,
        smooth_kernel=1,
        min_gap=0,
    )
    assert len(segs) == 1
    assert int(segs[0]["label"]) == 1
    assert float(segs[0]["start"]) == 1.0
    assert float(segs[0]["end"]) == 4.0


def test_decode_modes_both_work() -> None:
    rng = np.random.default_rng(0)
    probs = rng.uniform(size=(20, 4)).astype(np.float32)
    probs = probs / np.sum(probs, axis=1, keepdims=True)

    segs_a = decode_per_class_threshold(probs=probs, score_threshold=0.3, min_len=2, background_class=0)
    segs_b = decode_argmax_probs(
        probs=probs,
        score_threshold=0.3,
        min_len=2,
        background_class=0,
        smooth_kernel=3,
        min_gap=1,
    )
    assert isinstance(segs_a, list)
    assert isinstance(segs_b, list)
