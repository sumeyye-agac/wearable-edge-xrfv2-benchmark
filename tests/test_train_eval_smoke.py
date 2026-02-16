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
