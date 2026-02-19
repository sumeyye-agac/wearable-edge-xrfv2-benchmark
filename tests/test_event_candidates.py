from __future__ import annotations

import numpy as np

from xrfv2_edge_tal.event.candidates import (
    detect_candidates,
    motion_energy_from_glasses,
    widen_window,
)


def test_motion_energy_uses_gyro_when_available() -> None:
    # [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
    x_gl = np.array(
        [
            [0.0, 0.0, 0.0, 3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    energy = motion_energy_from_glasses(x_gl)
    assert energy.shape == (2,)
    assert abs(float(energy[0]) - 5.0) < 1e-6
    assert abs(float(energy[1])) < 1e-6


def test_detect_candidates_min_active_and_cooldown() -> None:
    energy = np.array([0.0, 1.1, 1.2, 0.0, 1.3, 1.3, 0.0], dtype=np.float32)
    runs = detect_candidates(
        energy=energy,
        thr=1.0,
        min_active_s=0.2,
        cooldown_s=0.3,
        frame_time_s=0.1,
    )
    # Two nearby runs merge due to cooldown.
    assert runs == [(1, 6)]


def test_widen_window_respects_boundaries() -> None:
    start, end = widen_window(
        start=2,
        end=4,
        pre_s=0.5,
        post_s=0.5,
        t_frames=6,
        frame_time_s=0.1,
    )
    assert start == 0
    assert end == 6
