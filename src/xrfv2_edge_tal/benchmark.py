"""Edge benchmarking utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from xrfv2_edge_tal.artifacts import create_run_dir, write_metrics
from xrfv2_edge_tal.checkpoint import load_checkpoint
from xrfv2_edge_tal.modalities import resolve_modalities_to_raw_keys
from xrfv2_edge_tal.models.factory import build_model


def _build_model_from_checkpoint(state: dict[str, Any], metadata: dict[str, Any], seed: int):
    model = build_model(
        name=str(metadata["model_name"]),
        input_dims=dict(metadata["input_dims"]),
        num_classes=int(metadata["num_classes"]),
        hidden_dim=int(metadata["hidden_dim"]),
        seed=seed,
        backend=str(metadata.get("backend", "numpy")),
        device=str(metadata.get("device", "auto")),
        kernel_size=int(state.get("kernel_size", 5)),
        tcn_layers=int(state.get("tcn_layers", metadata.get("tcn_layers", 1))),
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
) -> dict[str, Any]:
    for _ in range(warmup):
        _ = model.predict_proba(x_dict)

    times_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model.predict_proba(x_dict)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    return {
        "samples_ms": [float(x) for x in times_ms],
        "median": float(np.median(times_ms)) if times_ms else 0.0,
        "mean": float(np.mean(times_ms)) if times_ms else 0.0,
        "std": float(np.std(times_ms)) if times_ms else 0.0,
        "p90": float(np.percentile(times_ms, 90)) if times_ms else 0.0,
        "p95": float(np.percentile(times_ms, 95)) if times_ms else 0.0,
        "min": float(np.min(times_ms)) if times_ms else 0.0,
        "max": float(np.max(times_ms)) if times_ms else 0.0,
    }


def _input_dims_for_profile(
    input_dims: dict[str, int], config: dict[str, Any], profile: str | None
) -> dict[str, int]:
    if profile is None:
        return dict(input_dims)
    profile_map = config.get("data", {}).get("profiles", {})
    if profile not in profile_map:
        available = ", ".join(sorted(str(k) for k in profile_map.keys()))
        raise ValueError(f"Unknown profile '{profile}'. Available: {available}")
    requested = [str(item) for item in profile_map[profile]]
    raw_keys = resolve_modalities_to_raw_keys(
        available_modalities=input_dims.keys(), requested_modalities=requested
    )
    filtered = {k: input_dims[k] for k in raw_keys if k in input_dims}
    if not filtered:
        raise ValueError(
            f"Profile '{profile}' has no overlap with checkpoint modalities {list(input_dims)}"
        )
    return filtered


def benchmark_main(
    checkpoint: str,
    config: dict[str, Any],
    seed: int = 42,
    output_dir: str = "runs",
    profile: str | None = None,
) -> Path:
    state, metadata = load_checkpoint(checkpoint)
    model = _build_model_from_checkpoint(state=state, metadata=metadata, seed=seed)

    bench_cfg = config.get("benchmark", {})
    seq_len = int(bench_cfg.get("seq_len", 160))
    warmup = int(bench_cfg.get("warmup", 5))
    iters = int(bench_cfg.get("iters", 25))

    input_dims = _input_dims_for_profile(metadata["input_dims"], config=config, profile=profile)
    x = _make_fixed_input(input_dims, seq_len=seq_len, seed=seed)
    latency = _latency_stats(model=model, x_dict=x, warmup=warmup, iters=iters)

    checkpoint_size_mb = Path(checkpoint).stat().st_size / (1024.0 * 1024.0)
    params = _count_params(state)

    payload = {
        "profile": profile,
        "modalities": sorted(input_dims.keys()),
        "params": params,
        "checkpoint_size_mb": float(checkpoint_size_mb),
        "estimated_fps_median": float(1000.0 / max(latency["median"], 1e-9)),
        "estimated_fps_p90": float(1000.0 / max(latency["p90"], 1e-9)),
        "latency_budget_pass_50ms": bool(latency["p90"] <= 50.0),
        "edge_score_simple": float(
            (1.0 / max(latency["p90"], 1e-6)) * (1.0 / max(checkpoint_size_mb, 1e-6))
        ),
        "cpu_latency_ms": {
            **latency,
            "warmup": warmup,
            "iters": iters,
            "seq_len": seq_len,
        },
    }

    run_dir = create_run_dir(
        base_dir=output_dir, config_dict=config, command_str="xrfv2-edge-tal benchmark"
    )
    (run_dir / "benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "dataset_fingerprint.json").write_text("{}\n", encoding="utf-8")
    write_metrics(run_dir, {"benchmark": payload})

    return run_dir
