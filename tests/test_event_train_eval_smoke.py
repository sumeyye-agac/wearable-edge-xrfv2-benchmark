from __future__ import annotations

import json
from pathlib import Path

from xrfv2_edge_tal.event.eval_event import eval_event_main
from xrfv2_edge_tal.event.train_event import train_event_main


def _event_config() -> dict:
    return {
        "data": {
            "profiles": {
                "earbuds_glasses": ["earbuds", "glasses"],
                "glasses_only": ["glasses"],
                "all_imu": ["phone", "watch", "earbuds", "glasses"],
            },
            "default_profile": "earbuds_glasses",
        },
        "labels": {
            "positive_action_names": ["Answer the phone", "Use phone"],
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
            "modality_dropout_p": 0.05,
            "max_train_samples": 4,
        },
        "eval": {
            "split": "test",
            "max_eval_samples": 3,
            "frame_time_s": 0.02,
            "onset_tolerance_s": 0.4,
            "trigger": {
                "threshold": 0.5,
                "smoothing_window": 3,
                "cooldown_s": 0.2,
                "hysteresis": False,
            },
        },
    }


def test_event_train_eval_multi_profile_smoke(tmp_path: Path) -> None:
    cfg = _event_config()

    train_run = train_event_main(
        config=cfg,
        data_root=str(tmp_path / "raw"),
        adapter_name="dummy",
        seed=9,
        runs_dir=str(tmp_path / "runs"),
        profile="earbuds_glasses",
    )
    ckpt = train_run / "checkpoints" / "last.npz"
    assert ckpt.exists()

    eval_run = eval_event_main(
        checkpoint=str(ckpt),
        config=cfg,
        data_root=str(tmp_path / "raw"),
        adapter_name="dummy",
        seed=9,
        output_dir=str(tmp_path / "runs"),
        profile="earbuds_glasses",
        profiles=["earbuds_glasses", "glasses_only"],
    )

    metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
    assert "event_metrics" in metrics
    assert "profile_metrics" in metrics
    assert "earbuds_glasses" in metrics["profile_metrics"]
    assert "glasses_only" in metrics["profile_metrics"]

    report_path = eval_run / "profile_report.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "earbuds_glasses" in report_text
    assert "glasses_only" in report_text


def test_event_train_with_distillation_smoke(tmp_path: Path) -> None:
    teacher_cfg = _event_config()
    teacher_cfg["train"]["max_train_samples"] = 3

    teacher_run = train_event_main(
        config=teacher_cfg,
        data_root=str(tmp_path / "raw"),
        adapter_name="dummy",
        seed=13,
        runs_dir=str(tmp_path / "runs"),
        profile="all_imu",
    )
    teacher_ckpt = teacher_run / "checkpoints" / "last.npz"
    assert teacher_ckpt.exists()

    student_cfg = _event_config()
    student_cfg["train"]["distillation"] = {
        "enabled": True,
        "teacher_checkpoint": str(teacher_ckpt),
        "weight": 0.25,
        "temperature": 2.0,
    }
    student_cfg["train"]["max_train_samples"] = 3

    student_run = train_event_main(
        config=student_cfg,
        data_root=str(tmp_path / "raw"),
        adapter_name="dummy",
        seed=13,
        runs_dir=str(tmp_path / "runs"),
        profile="earbuds_glasses",
    )
    metrics = json.loads((student_run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["distillation"]["enabled"] is True


def test_event_eval_uses_checkpoint_tcn_shape_over_config(tmp_path: Path) -> None:
    train_cfg = _event_config()
    train_cfg["model"]["kernel_size"] = 7
    train_cfg["model"]["tcn_layers"] = 2
    train_cfg["train"]["max_train_samples"] = 3

    train_run = train_event_main(
        config=train_cfg,
        data_root=str(tmp_path / "raw"),
        adapter_name="dummy",
        seed=21,
        runs_dir=str(tmp_path / "runs"),
        profile="earbuds_glasses",
    )
    ckpt = train_run / "checkpoints" / "last.npz"
    assert ckpt.exists()

    eval_cfg = _event_config()
    eval_cfg["model"]["kernel_size"] = 3
    eval_cfg["model"]["tcn_layers"] = 1

    eval_run = eval_event_main(
        checkpoint=str(ckpt),
        config=eval_cfg,
        data_root=str(tmp_path / "raw"),
        adapter_name="dummy",
        seed=21,
        output_dir=str(tmp_path / "runs"),
        profile="earbuds_glasses",
        profiles=["earbuds_glasses"],
    )
    metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
    assert "profile_metrics" in metrics
