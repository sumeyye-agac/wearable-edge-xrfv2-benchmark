from __future__ import annotations

import numpy as np

from xrfv2_edge_tal.paper_track import (
    aggregate_window_probs,
    make_windows,
    resample_sample,
    to_frame_segments,
)


def test_make_windows_and_aggregate() -> None:
    x = {"imu_phone": np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32)}
    segments = [{"start": 10.0, "end": 25.0, "label": 1}]
    windows = make_windows(x_dict=x, segments=segments, clip_len=32, stride=16, min_coverage=0.25)
    assert windows
    assert all(w["x"]["imu_phone"].shape == (32, 3) for w in windows)

    probs = [np.full((32, 4), 0.25, dtype=np.float32) for _ in windows]
    agg = aggregate_window_probs(
        window_probs=probs,
        starts=[int(w["start"]) for w in windows],
        valid_lens=[int(w["valid_len"]) for w in windows],
        full_len=100,
    )
    assert agg.shape == (100, 4)
    assert np.allclose(np.sum(agg, axis=1), 1.0, atol=1e-6)


def test_resample_and_normalized_segments() -> None:
    x = {"imu": np.random.default_rng(1).normal(size=(50, 2)).astype(np.float32)}
    normalized_segments = [{"start": 0.2, "end": 0.5, "label": 2}]
    frame_segments = to_frame_segments(normalized_segments, seq_len=50)
    xr, sr = resample_sample(x_dict=x, segments=frame_segments, target_len=100)
    assert xr["imu"].shape == (100, 2)
    assert 19.0 <= float(sr[0]["start"]) <= 21.0
    assert 49.0 <= float(sr[0]["end"]) <= 51.0
