from __future__ import annotations

import json
from pathlib import Path

from xrfv2_edge_tal.eval import eval_main
from xrfv2_edge_tal.train import train_main


def test_train_eval_dummy_smoke(tmp_path: Path) -> None:
    config = {
        "model": {
            "name": "tiny_tcn",
            "num_classes": 5,
            "hidden_dim": 16,
            "kernel_size": 3,
        },
        "train": {
            "epochs": 1,
            "lr": 0.02,
            "modality_dropout_p": 0.1,
        },
        "decode": {
            "score_threshold": 0.5,
            "min_len": 2,
            "nms_tiou": 0.5,
        },
        "eval": {
            "split": "test",
        },
    }

    train_run = train_main(
        config=config,
        data_root=str(tmp_path / "data_root"),
        adapter_name="dummy",
        seed=7,
        runs_dir=str(tmp_path / "runs"),
    )
    ckpt = train_run / "checkpoints" / "last.npz"
    assert ckpt.exists()

    eval_run = eval_main(
        checkpoint=str(ckpt),
        config=config,
        data_root=str(tmp_path / "data_root"),
        adapter_name="dummy",
        seed=7,
        output_dir=str(tmp_path / "runs"),
    )
    metrics_path = eval_run / "metrics.json"
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "eval" in metrics
    assert "map_avg" in metrics["eval"]


def test_train_eval_dummy_paper_track_smoke(tmp_path: Path) -> None:
    config = {
        "model": {
            "name": "tiny_tcn",
            "num_classes": 5,
            "hidden_dim": 12,
            "kernel_size": 3,
        },
        "train": {
            "epochs": 1,
            "lr": 0.01,
            "lr_schedule": "cosine",
            "paper_track": {
                "enabled": True,
                "clip_len": 64,
                "stride": 32,
                "min_segment_coverage": 0.25,
                "resample_to": 96,
                "max_windows_per_sample": 2,
                "noise_std": 0.0,
                "scale_jitter": 0.0,
            },
        },
        "decode": {
            "score_threshold": 0.4,
            "min_len": 2,
            "nms_tiou": 0.5,
            "background_class": 0,
        },
        "eval": {
            "split": "test",
            "max_eval_samples": 3,
            "paper_track": {
                "enabled": True,
                "clip_len": 64,
                "stride": 32,
                "min_segment_coverage": 0.25,
                "resample_to": 96,
            },
        },
    }

    train_run = train_main(
        config=config,
        data_root=str(tmp_path / "data_root"),
        adapter_name="dummy",
        seed=11,
        runs_dir=str(tmp_path / "runs"),
    )
    ckpt = train_run / "checkpoints" / "last.npz"
    assert ckpt.exists()

    eval_run = eval_main(
        checkpoint=str(ckpt),
        config=config,
        data_root=str(tmp_path / "data_root"),
        adapter_name="dummy",
        seed=11,
        output_dir=str(tmp_path / "runs"),
    )
    metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
    assert "paper_track" in metrics
    assert metrics["paper_track"]["enabled"] is True


def test_train_eval_dummy_with_selected_modality_and_argmax_decode(tmp_path: Path) -> None:
    config = {
        "data": {
            "modalities": ["imu_watch"],
        },
        "model": {
            "name": "tiny_tcn",
            "num_classes": 5,
            "hidden_dim": 12,
            "kernel_size": 3,
        },
        "train": {
            "epochs": 1,
            "lr": 0.01,
        },
        "decode": {
            "mode": "argmax",
            "score_threshold": 0.3,
            "min_len": 2,
            "nms_tiou": 0.5,
            "background_class": 0,
            "smooth_kernel": 3,
            "min_gap": 1,
        },
        "eval": {
            "split": "test",
            "max_eval_samples": 4,
        },
    }

    train_run = train_main(
        config=config,
        data_root=str(tmp_path / "data_root"),
        adapter_name="dummy",
        seed=5,
        runs_dir=str(tmp_path / "runs"),
    )
    ckpt = train_run / "checkpoints" / "last.npz"
    assert ckpt.exists()

    eval_run = eval_main(
        checkpoint=str(ckpt),
        config=config,
        data_root=str(tmp_path / "data_root"),
        adapter_name="dummy",
        seed=5,
        output_dir=str(tmp_path / "runs"),
    )
    metrics = json.loads((eval_run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["decode"]["mode"] == "argmax"
    assert metrics["selected_modalities"] == ["imu_watch"]
