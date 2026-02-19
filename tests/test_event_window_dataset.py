from __future__ import annotations

import numpy as np

from xrfv2_edge_tal.event.window_dataset import build_windows_for_sample


def test_build_windows_for_sample_labels_overlap() -> None:
    t = 100
    x = {
        "imu_gl": np.zeros((t, 6), dtype=np.float32),
        "airpods": np.zeros((t, 6), dtype=np.float32),
    }
    # Build a clear motion region in gyro channels.
    x["imu_gl"][30:45, 3:6] = 2.0

    segments = [{"label": 5, "start": 0.32, "end": 0.42}]
    windows = build_windows_for_sample(
        x_dict=x,
        segments=segments,
        positive_ids={5},
        profile_modalities=["airpods", "imu_gl"],
        frame_time_s=0.02,
        candidate_cfg={
            "energy_threshold": 1.0,
            "min_active_s": 0.1,
            "cooldown_s": 0.1,
            "window_len_s": 1.0,
            "overlap_min_s": 0.1,
            "max_windows": 10,
            "include_gt_windows": False,
        },
        window_len_s=1.0,
        sample_id="s0",
    )

    assert windows
    assert any(int(w["y"]) == 1 for w in windows)
    first = windows[0]
    assert set(first["x_window"].keys()) == {"airpods", "imu_gl"}
    assert first["x_window"]["imu_gl"].shape[0] == first["x_window"]["airpods"].shape[0]


def test_build_windows_handles_missing_glasses() -> None:
    x = {"airpods": np.zeros((64, 6), dtype=np.float32)}
    windows = build_windows_for_sample(
        x_dict=x,
        segments=[],
        positive_ids={1},
        profile_modalities=["airpods"],
        frame_time_s=0.02,
        candidate_cfg={},
        window_len_s=1.0,
    )
    assert windows == []
