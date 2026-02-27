from __future__ import annotations

import json
from pathlib import Path

from xrfv2_edge_tal.event.calibrate_event import calibrate_event_main
from xrfv2_edge_tal.event.train_event import train_event_main


def _config() -> dict:
    return {
        "data": {
            "profiles": {
                "earbuds_glasses": ["earbuds", "glasses"],
                "all_imu": ["phone", "watch", "earbuds", "glasses"],
            },
            "default_profile": "earbuds_glasses",
            "normalize_per_sample": True,
            "normalize_clip": 8.0,
        },
        "labels": {
            "positive_action_names": ["Answer the phone", "Use phone"],
            "source_modality": "imu",
        },
        "model": {
            "name": "tiny_tcn",
            "hidden_dim": 12,
            "kernel_size": 3,
            "tcn_layers": 1,
        },
        "runtime": {
            "backend": "torch",
            "device": "cpu",
        },
        "train": {
            "epochs": 1,
            "lr": 0.01,
            "max_train_samples": 6,
        },
        "eval": {
            "split": "test",
            "max_eval_samples": 4,
            "frame_time_s": 0.02,
            "onset_tolerance_s": 0.5,
            "trigger": {
                "threshold": 0.5,
                "smoothing_window": 3,
                "cooldown_s": 0.2,
                "hysteresis": False,
                "threshold_off": 0.2,
            },
        },
    }


def test_event_calibrate_smoke(tmp_path: Path) -> None:
    cfg = _config()
    train_run = train_event_main(
        config=cfg,
        data_root=str(tmp_path / "raw"),
        adapter_name="dummy",
        seed=7,
        runs_dir=str(tmp_path / "runs"),
        profile="earbuds_glasses",
    )
    ckpt = train_run / "checkpoints" / "last.npz"
    assert ckpt.exists()

    cal_run = calibrate_event_main(
        checkpoint=str(ckpt),
        config=cfg,
        data_root=str(tmp_path / "raw"),
        adapter_name="dummy",
        seed=7,
        output_dir=str(tmp_path / "runs"),
        profiles=["earbuds_glasses", "all_imu"],
        thresholds=[0.3, 0.5],
        cooldowns=[0.1, 0.2],
        metric_mode="within_segment",
    )

    assert (cal_run / "calibration_grid.json").exists()
    assert (cal_run / "calibration_report.md").exists()

    metrics = json.loads((cal_run / "metrics.json").read_text(encoding="utf-8"))
    assert "calibration" in metrics
    best = metrics["calibration"]["best_by_profile"]
    assert "earbuds_glasses" in best
    assert "all_imu" in best


def test_event_calibrate_sample_presence_mode(tmp_path: Path) -> None:
    cfg = _config()
    train_run = train_event_main(
        config=cfg,
        data_root=str(tmp_path / "raw"),
        adapter_name="dummy",
        seed=11,
        runs_dir=str(tmp_path / "runs"),
        profile="earbuds_glasses",
    )
    ckpt = train_run / "checkpoints" / "last.npz"
    assert ckpt.exists()

    cal_run = calibrate_event_main(
        checkpoint=str(ckpt),
        config=cfg,
        data_root=str(tmp_path / "raw"),
        adapter_name="dummy",
        seed=11,
        output_dir=str(tmp_path / "runs"),
        profiles=["earbuds_glasses"],
        thresholds=[0.3, 0.5],
        cooldowns=[0.1, 0.2],
        metric_mode="sample_presence",
    )
    metrics = json.loads((cal_run / "metrics.json").read_text(encoding="utf-8"))
    best = metrics["calibration"]["best_by_profile"]["earbuds_glasses"]["best_row"]
    assert "sample_presence" in best
