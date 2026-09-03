from __future__ import annotations

import json
from pathlib import Path

from xrfv2_edge_tal.benchmark import _input_dims_for_profile, benchmark_main
from xrfv2_edge_tal.train import train_main


def test_input_dims_for_profile_resolves_aliased_raw_keys() -> None:
    # Checkpoint trained via the dummy adapter uses its own raw modality names,
    # while the shipped config's profile lists use the real-dataset raw names.
    # Both must resolve to the same canonical modality for filtering to work.
    input_dims = {"imu_earbuds": 6, "imu_glasses": 6}
    config = {
        "data": {
            "profiles": {
                "earbuds_glasses": ["airpods", "imu_gl"],
                "glasses_only": ["imu_gl"],
            }
        }
    }
    assert _input_dims_for_profile(input_dims, config=config, profile="earbuds_glasses") == {
        "imu_earbuds": 6,
        "imu_glasses": 6,
    }
    assert _input_dims_for_profile(input_dims, config=config, profile="glasses_only") == {
        "imu_glasses": 6,
    }


def test_benchmark_dummy_smoke(tmp_path: Path) -> None:
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
        },
        "benchmark": {
            "seq_len": 64,
            "warmup": 1,
            "iters": 3,
        },
    }

    train_run = train_main(
        config=config,
        data_root=str(tmp_path / "data_root"),
        adapter_name="dummy",
        seed=3,
        runs_dir=str(tmp_path / "runs"),
    )

    checkpoint = train_run / "checkpoints" / "last.npz"
    assert checkpoint.exists()

    bench_run = benchmark_main(
        checkpoint=str(checkpoint),
        config=config,
        seed=3,
        output_dir=str(tmp_path / "runs"),
    )
    benchmark_path = bench_run / "benchmark.json"
    assert benchmark_path.exists()

    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert payload["params"] > 0
    assert payload["cpu_latency_ms"]["median"] >= 0.0
