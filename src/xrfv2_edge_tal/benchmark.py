"""Edge benchmarking utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from xrfv2_edge_tal.artifacts import create_run_dir, write_metrics
from xrfv2_edge_tal.checkpoint import load_checkpoint
from xrfv2_edge_tal.models.factory import build_model


def _build_model_from_checkpoint(state: dict[str, Any], metadata: dict[str, Any], seed: int):
    model = build_model(
        name=str(metadata["model_name"]),
        input_dims=dict(metadata["input_dims"]),
        num_classes=int(metadata["num_classes"]),
        hidden_dim=int(metadata["hidden_dim"]),
        seed=seed,
        kernel_size=int(state.get("kernel_size", 5)),
    )
    model.load_state_dict(state)
    return model


def _count_params(state: dict[str, Any]) -> int:
    total = 0
    for value in state.values():
        if isinstance(value, np.ndarray):
            total += int(value.size)
    return total


def _make_fixed_input(input_dims: dict[str, int], seq_len: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        modality: rng.normal(0.0, 1.0, size=(seq_len, dim)).astype(np.float32)
        for modality, dim in input_dims.items()
    }


def _latency_stats(
    model: Any,
    x_dict: dict[str, np.ndarray],
    warmup: int = 5,
    iters: int = 25,
) -> tuple[float, float]:
    for _ in range(warmup):
        _ = model.predict_proba(x_dict)

    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model.predict_proba(x_dict)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    median = float(np.median(times_ms)) if times_ms else 0.0
    p90 = float(np.percentile(times_ms, 90)) if times_ms else 0.0
    return median, p90


def benchmark_main(
    checkpoint: str,
    config: dict[str, Any],
    seed: int = 42,
    output_dir: str = "runs",
) -> Path:
    state, metadata = load_checkpoint(checkpoint)
    model = _build_model_from_checkpoint(state=state, metadata=metadata, seed=seed)

    bench_cfg = config.get("benchmark", {})
    seq_len = int(bench_cfg.get("seq_len", 160))
    warmup = int(bench_cfg.get("warmup", 5))
    iters = int(bench_cfg.get("iters", 25))

    x = _make_fixed_input(metadata["input_dims"], seq_len=seq_len, seed=seed)
    median_ms, p90_ms = _latency_stats(model=model, x_dict=x, warmup=warmup, iters=iters)

    checkpoint_size_mb = Path(checkpoint).stat().st_size / (1024.0 * 1024.0)
    params = _count_params(state)

    payload = {
        "params": params,
        "checkpoint_size_mb": float(checkpoint_size_mb),
        "cpu_latency_ms": {
            "median": median_ms,
            "p90": p90_ms,
            "warmup": warmup,
            "iters": iters,
            "seq_len": seq_len,
        },
    }

    run_dir = create_run_dir(base_dir=output_dir, config_dict=config, command_str="xrfv2-edge-tal benchmark")
    (run_dir / "benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "dataset_fingerprint.json").write_text("{}\n", encoding="utf-8")
    write_metrics(run_dir, {"benchmark": payload})

    return run_dir
