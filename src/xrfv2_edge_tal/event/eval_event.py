"""Evaluation entrypoint for phone-interaction event detection."""

from __future__ import annotations

import json
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from xrfv2_edge_tal.artifacts import create_run_dir, write_metrics
from xrfv2_edge_tal.checkpoint import load_checkpoint
from xrfv2_edge_tal.data.adapters import DummyAdapter, XRFV2H5Adapter
from xrfv2_edge_tal.event.candidates import (
    default_candidate_config,
    detect_candidates,
    motion_energy_from_glasses,
    widen_window,
)
from xrfv2_edge_tal.event.metrics import compute_event_metrics
from xrfv2_edge_tal.event.preprocess import normalization_config, normalize_modalities
from xrfv2_edge_tal.event.trigger import filter_trigger_candidates, frame_probs_to_event_triggers
from xrfv2_edge_tal.labels.xrfv2_labels import resolve_positive_label_ids, resolve_proxy_label_ids
from xrfv2_edge_tal.modalities import resolve_modalities_to_raw_keys
from xrfv2_edge_tal.models.factory import build_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _adapter_from_name(
    adapter_name: str, data_root: str, seed: int
) -> DummyAdapter | XRFV2H5Adapter:
    if adapter_name == "dummy":
        return DummyAdapter(seed=seed)
    if adapter_name == "xrfv2":
        return XRFV2H5Adapter(data_root)
    raise ValueError(f"Unsupported adapter: {adapter_name}")


def _resolve_profile_names(
    config: dict[str, Any], profile: str | None, profiles: list[str] | None
) -> list[str]:
    data_cfg = config.get("data", {})
    profile_map = data_cfg.get("profiles", {})
    if not isinstance(profile_map, dict) or not profile_map:
        raise ValueError("Config is missing data.profiles for event evaluation")

    if profiles:
        out: list[str] = []
        for item in profiles:
            key = str(item).strip()
            if not key:
                continue
            if key not in profile_map:
                available = ", ".join(sorted(str(k) for k in profile_map.keys()))
                raise ValueError(f"Unknown profile '{key}'. Available: {available}")
            if key not in out:
                out.append(key)
        if out:
            return out

    selected = profile or str(data_cfg.get("default_profile", "earbuds_glasses"))
    if selected not in profile_map:
        available = ", ".join(sorted(str(k) for k in profile_map.keys()))
        raise ValueError(f"Unknown profile '{selected}'. Available: {available}")
    return [selected]


def _resolve_positive_ids(config: dict[str, Any], adapter_name: str, data_root: str) -> set[int]:
    def _parse_explicit_ids(raw: Any) -> set[int]:
        if raw is None:
            return set()
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.replace(";", ",").split(",")]
            return {int(p) for p in parts if p}
        if isinstance(raw, (list, tuple, set)):
            out: set[int] = set()
            for item in raw:
                if isinstance(item, str) and "," in item:
                    out |= _parse_explicit_ids(item)
                elif str(item).strip():
                    out.add(int(item))
            return out
        return {int(raw)}

    labels_cfg = config.get("labels", {})
    task_variant = str(labels_cfg.get("task_variant", "phone_interaction")).strip().lower()
    explicit = _parse_explicit_ids(labels_cfg.get("positive_label_ids", []))
    names = labels_cfg.get("positive_action_names", ["Answer the phone", "Use phone"])
    proxy_keywords = labels_cfg.get("proxy_keywords", ["head", "face", "phone", "ear", "glasses"])

    if explicit:
        return explicit
    if adapter_name == "dummy":
        return {1, 2} if task_variant == "phone_interaction" else {1, 2, 3}
    if task_variant == "hand_to_head_proxy":
        return resolve_proxy_label_ids(
            data_root=data_root,
            keywords=[str(x) for x in proxy_keywords],
            fallback_positive_label_ids=None,
        )
    return resolve_positive_label_ids(
        data_root=data_root,
        positive_action_names=[str(x) for x in names],
        fallback_positive_label_ids=None,
    )


def _resolve_label_source_modality(config: dict[str, Any]) -> str:
    labels_cfg = config.get("labels", {})
    source = str(labels_cfg.get("source_modality", "imu")).strip()
    return source or "imu"


def _resolve_event_mode(config: dict[str, Any]) -> str:
    eval_cfg = config.get("eval", {})
    train_cfg = config.get("train", {})
    mode = str(eval_cfg.get("event_mode", train_cfg.get("event_mode", "flat"))).strip().lower()
    if mode not in {"flat", "hierarchical"}:
        raise ValueError(f"Unsupported eval.event_mode={mode}. Expected flat|hierarchical.")
    return mode


def _resolve_hierarchical_cfg(config: dict[str, Any]) -> dict[str, float | int]:
    eval_cfg = config.get("eval", {})
    raw = eval_cfg.get("hierarchical", {})
    if not isinstance(raw, dict):
        raw = {}
    return default_candidate_config(raw)


def _segments_for_sample(
    adapter: DummyAdapter | XRFV2H5Adapter,
    sample_id: str,
    split: str,
    fallback_segments: list[dict[str, float | int]],
    source_modality: str,
) -> list[dict[str, float | int]]:
    if hasattr(adapter, "get_segments"):
        return adapter.get_segments(  # type: ignore[attr-defined]
            sample_id=sample_id,
            split=split,
            source_modality=source_modality,
        )
    return fallback_segments


def _select_modalities_for_profile(
    x: dict[str, np.ndarray],
    config: dict[str, Any],
    profile_name: str,
) -> dict[str, np.ndarray]:
    requested = config["data"]["profiles"][profile_name]
    raw_keys = resolve_modalities_to_raw_keys(
        available_modalities=x.keys(),
        requested_modalities=[str(item) for item in requested],
    )
    return {key: x[key] for key in raw_keys if key in x}


def _segment_start_seconds(seg: dict[str, Any], seq_len: int, frame_time_s: float) -> float:
    raw_start = float(seg["start"])
    if raw_start <= 1.0:
        raw_start *= seq_len
    return float(raw_start * frame_time_s)


def _segment_end_seconds(seg: dict[str, Any], seq_len: int, frame_time_s: float) -> float:
    raw_end = float(seg["end"])
    if raw_end <= 1.0:
        raw_end *= seq_len
    return float(raw_end * frame_time_s)


def _sequence_duration_seconds(meta: dict[str, Any], seq_len: int, frame_time_s: float) -> float:
    if "duration_s" in meta:
        return float(meta["duration_s"])
    if "timestamps" in meta:
        stamps = meta["timestamps"]
        if isinstance(stamps, list) and len(stamps) >= 2:
            return float(stamps[-1] - stamps[0])
    return float(seq_len * frame_time_s)


def _count_params(state: dict[str, Any]) -> int:
    total = 0
    for value in state.values():
        if isinstance(value, np.ndarray):
            total += int(value.size)
    return total


def _window_frames(window_len_s: float, frame_time_s: float) -> int:
    raw = int(np.round(max(0.1, float(window_len_s)) / max(frame_time_s, 1e-9)))
    return int(np.clip(raw, 16, 512))


def _slice_or_pad_window(arr: np.ndarray, start: int, end: int, target_len: int) -> np.ndarray:
    clip_start = max(0, int(start))
    clip_end = min(int(arr.shape[0]), int(end))
    if clip_end <= clip_start:
        clip_end = min(int(arr.shape[0]), clip_start + 1)
    part = arr[clip_start:clip_end]
    if part.shape[0] == target_len:
        return part.astype(np.float32, copy=False)
    if part.shape[0] > target_len:
        return part[:target_len].astype(np.float32, copy=False)
    pad_rows = target_len - part.shape[0]
    if part.shape[0] == 0:
        base = np.zeros((1, arr.shape[1]), dtype=np.float32)
    else:
        base = part[-1:, :].astype(np.float32, copy=False)
    pad = np.repeat(base, pad_rows, axis=0)
    return np.concatenate([part.astype(np.float32, copy=False), pad], axis=0)


def _fixed_window(start: int, end: int, seq_len: int, w_frames: int) -> tuple[int, int]:
    center = int((start + end) // 2)
    out_start = center - (w_frames // 2)
    out_end = out_start + w_frames
    if out_start < 0:
        out_end += -out_start
        out_start = 0
    if out_end > seq_len:
        shift = out_end - seq_len
        out_start = max(0, out_start - shift)
        out_end = seq_len
    if out_end - out_start < w_frames:
        out_start = max(0, out_end - w_frames)
    return int(out_start), int(min(seq_len, out_start + w_frames))


def _hierarchical_candidates(
    x_profile: dict[str, np.ndarray],
    frame_time_s: float,
    hierarchical_cfg: dict[str, float | int],
) -> list[tuple[int, int]]:
    if "imu_gl" not in x_profile:
        return []
    seq_len = int(x_profile["imu_gl"].shape[0])
    if seq_len <= 0:
        return []
    energy = motion_energy_from_glasses(x_profile["imu_gl"])
    raw = detect_candidates(
        energy=energy,
        thr=float(hierarchical_cfg["energy_threshold"]),
        min_active_s=float(hierarchical_cfg["min_active_s"]),
        cooldown_s=float(hierarchical_cfg["cooldown_s"]),
        frame_time_s=frame_time_s,
    )
    w_frames = _window_frames(float(hierarchical_cfg["window_len_s"]), frame_time_s)
    out: list[tuple[int, int]] = []
    seen: set[int] = set()
    for start, end in raw:
        s, e = widen_window(
            start=start,
            end=end,
            pre_s=float(hierarchical_cfg["pre_s"]),
            post_s=float(hierarchical_cfg["post_s"]),
            t_frames=seq_len,
            frame_time_s=frame_time_s,
        )
        ws, we = _fixed_window(s, e, seq_len=seq_len, w_frames=w_frames)
        if ws not in seen:
            out.append((ws, we))
            seen.add(ws)
    max_windows = max(1, int(hierarchical_cfg["max_windows"]))
    return out[:max_windows]


def _latency_stats(
    model: Any,
    x_dict: dict[str, np.ndarray],
    warmup: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup):
        _ = model.predict_proba(x_dict)

    samples_ms: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = model.predict_proba(x_dict)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "median": float(np.median(samples_ms)) if samples_ms else 0.0,
        "p90": float(np.percentile(samples_ms, 90)) if samples_ms else 0.0,
    }


def _evaluate_profile(
    model: Any,
    adapter: DummyAdapter | XRFV2H5Adapter,
    split: str,
    profile_name: str,
    config: dict[str, Any],
    positive_ids: set[int],
    source_modality: str,
    event_mode: str,
    hierarchical_cfg: dict[str, float | int],
    norm_enabled: bool,
    norm_clip: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], float]:
    eval_cfg = config.get("eval", {})
    trigger_cfg = eval_cfg.get("trigger", {})

    frame_time_s = float(eval_cfg.get("frame_time_s", 0.02))
    onset_tolerance_s = float(eval_cfg.get("onset_tolerance_s", 0.5))
    max_eval_samples = int(eval_cfg.get("max_eval_samples", 0))

    threshold = float(trigger_cfg.get("threshold", 0.55))
    smoothing_window = int(trigger_cfg.get("smoothing_window", 5))
    cooldown_s = float(trigger_cfg.get("cooldown_s", 0.75))
    hysteresis = bool(trigger_cfg.get("hysteresis", False))
    threshold_off = trigger_cfg.get("threshold_off", None)
    min_active_s = float(trigger_cfg.get("min_active_s", 0.0))

    sample_ids = adapter.split_ids(split)
    if max_eval_samples > 0:
        sample_ids = sample_ids[:max_eval_samples]

    predictions: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    duration_s = 0.0

    for sample_id in sample_ids:
        x, segments_raw, meta = adapter.get_sample(sample_id, split)
        segments = _segments_for_sample(
            adapter=adapter,
            sample_id=sample_id,
            split=split,
            fallback_segments=segments_raw,
            source_modality=source_modality,
        )
        x_sel = _select_modalities_for_profile(x, config=config, profile_name=profile_name)
        x_sel = normalize_modalities(x_sel, enabled=norm_enabled, clip=norm_clip)
        seq_len = int(next(iter(x_sel.values())).shape[0])
        if event_mode == "flat":
            probs = model.predict_proba(x_sel)
            if probs.ndim != 2 or probs.shape[1] < 2:
                raise ValueError(f"Expected model output [T,2+] for event task, got {probs.shape}")
            pos_probs = probs[:, 1]
            triggers = frame_probs_to_event_triggers(
                probs=pos_probs,
                frame_time_s=frame_time_s,
                threshold=threshold,
                smoothing_window=smoothing_window,
                cooldown_s=cooldown_s,
                hysteresis=hysteresis,
                threshold_off=float(threshold_off) if threshold_off is not None else None,
                min_active_s=min_active_s,
            )
        else:
            candidates = _hierarchical_candidates(
                x_profile=x_sel,
                frame_time_s=frame_time_s,
                hierarchical_cfg=hierarchical_cfg,
            )
            w_frames = _window_frames(float(hierarchical_cfg["window_len_s"]), frame_time_s)
            scored: list[dict[str, float | int]] = []
            for start_frame, end_frame in candidates:
                x_window = {
                    key: _slice_or_pad_window(arr, start_frame, end_frame, w_frames)
                    for key, arr in x_sel.items()
                }
                probs = model.predict_proba(x_window)
                if probs.ndim != 2 or probs.shape[1] < 2:
                    raise ValueError(
                        f"Expected model output [W,2+] for event task, got {probs.shape}"
                    )
                score = float(np.mean(probs[:, 1]))
                scored.append(
                    {
                        "time": float(start_frame * frame_time_s),
                        "score": score,
                        "frame": int(start_frame),
                    }
                )
            triggers = filter_trigger_candidates(
                candidates=scored,
                threshold=threshold,
                cooldown_s=cooldown_s,
                hysteresis=hysteresis,
                threshold_off=float(threshold_off) if threshold_off is not None else None,
            )

        for item in triggers:
            predictions.append(
                {
                    "sample_id": sample_id,
                    "time": float(item["time"]),
                    "score": float(item["score"]),
                    "frame": int(item["frame"]),
                    "profile": profile_name,
                }
            )

        for seg in segments:
            if int(seg["label"]) in positive_ids:
                ground_truth.append(
                    {
                        "sample_id": sample_id,
                        "start": _segment_start_seconds(
                            seg, seq_len=seq_len, frame_time_s=frame_time_s
                        ),
                        "end": _segment_end_seconds(
                            seg, seq_len=seq_len, frame_time_s=frame_time_s
                        ),
                    }
                )

        duration_s += _sequence_duration_seconds(
            meta=meta, seq_len=seq_len, frame_time_s=frame_time_s
        )

    metrics = compute_event_metrics(
        predictions=predictions,
        ground_truth=ground_truth,
        duration_s=duration_s,
        onset_tolerance_s=onset_tolerance_s,
    )
    metrics["num_predictions"] = len(predictions)
    metrics["num_ground_truth"] = len(ground_truth)
    return metrics, predictions, ground_truth, duration_s


def _profile_note(profile_name: str) -> str:
    if profile_name == "wifi_all":
        return "upper-bound profile with Wi-Fi + all IMU; not product-realistic"
    if profile_name == "glasses_only":
        return "fallback profile: expected drop; used for robustness"
    if profile_name == "all_imu":
        return "diagnostic upper bound; non-product profile"
    return "default product target"


def _render_profile_report(
    profile_metrics: OrderedDict[str, dict[str, Any]],
    config: dict[str, Any],
    event_mode: str,
) -> str:
    lines = [
        "# Event Profile Report",
        "",
        f"- Event mode: `{event_mode}`",
        "",
        "| Profile | Sensors | Onset F1 | Within F1 | Onset FP/hour | Within FP/hour | p90 onset delay (s) | Notes |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for profile_name, metrics in profile_metrics.items():
        sensors = ", ".join(config["data"]["profiles"][profile_name])
        onset = metrics.get("onset_strict", {})
        within = metrics.get("within_segment", {})
        p90 = float(onset.get("onset_delay_s", {}).get("p90", 0.0))
        lines.append(
            "| "
            f"`{profile_name}` | {sensors} | "
            f"{float(onset.get('f1', 0.0)):.4f} | {float(within.get('f1', 0.0)):.4f} | "
            f"{float(onset.get('fp_per_hour', 0.0)):.3f} | {float(within.get('fp_per_hour', 0.0)):.3f} | "
            f"{p90:.3f} | {_profile_note(profile_name)} |"
        )
    lines.append("")
    return "\n".join(lines)


def eval_event_main(
    checkpoint: str,
    config: dict[str, Any],
    data_root: str,
    adapter_name: str = "dummy",
    seed: int = 42,
    output_dir: str = "runs",
    profile: str | None = None,
    profiles: list[str] | None = None,
) -> Path:
    _set_seed(seed)

    state, metadata = load_checkpoint(checkpoint)
    runtime_cfg = config.get("runtime", {})
    model_cfg = config.get("model", {})
    eval_cfg = config.get("eval", {})
    data_cfg = config.get("data", {})

    model = build_model(
        name=str(metadata.get("model_name", model_cfg.get("name", "tiny_tcn"))),
        input_dims=dict(metadata.get("input_dims", {})),
        num_classes=int(metadata.get("num_classes", 2)),
        hidden_dim=int(metadata.get("hidden_dim", model_cfg.get("hidden_dim", 24))),
        seed=seed,
        backend=str(metadata.get("backend", runtime_cfg.get("backend", "torch"))),
        device=str(metadata.get("device", runtime_cfg.get("device", "auto"))),
        kernel_size=int(state.get("kernel_size", model_cfg.get("kernel_size", 5))),
        tcn_layers=int(
            state.get("tcn_layers", model_cfg.get("tcn_layers", metadata.get("tcn_layers", 1)))
        ),
    )
    model.load_state_dict(state)

    adapter = _adapter_from_name(adapter_name=adapter_name, data_root=data_root, seed=seed)
    positive_ids = _resolve_positive_ids(
        config=config, adapter_name=adapter_name, data_root=data_root
    )
    source_modality = _resolve_label_source_modality(config)
    event_mode = _resolve_event_mode(config)
    hierarchical_cfg = _resolve_hierarchical_cfg(config)
    norm_enabled, norm_clip = normalization_config(data_cfg)

    split = str(eval_cfg.get("split", "test"))
    selected_profiles = _resolve_profile_names(config=config, profile=profile, profiles=profiles)

    profile_metrics: OrderedDict[str, dict[str, Any]] = OrderedDict()
    all_predictions: list[dict[str, Any]] = []
    all_ground_truth: list[dict[str, Any]] = []

    warmup = int(eval_cfg.get("benchmark_warmup", 5))
    iters = int(eval_cfg.get("benchmark_iters", 20))
    checkpoint_size_mb = Path(checkpoint).stat().st_size / (1024.0 * 1024.0)
    params = _count_params(state)

    for profile_name in selected_profiles:
        metrics, predictions, ground_truth, _ = _evaluate_profile(
            model=model,
            adapter=adapter,
            split=split,
            profile_name=profile_name,
            config=config,
            positive_ids=positive_ids,
            source_modality=source_modality,
            event_mode=event_mode,
            hierarchical_cfg=hierarchical_cfg,
            norm_enabled=norm_enabled,
            norm_clip=norm_clip,
        )

        sample_x, _, _ = adapter.get_sample(adapter.split_ids(split)[0], split)
        sample_x_sel = _select_modalities_for_profile(
            sample_x, config=config, profile_name=profile_name
        )
        sample_x_sel = normalize_modalities(sample_x_sel, enabled=norm_enabled, clip=norm_clip)
        if event_mode == "hierarchical":
            candidates = _hierarchical_candidates(
                x_profile=sample_x_sel,
                frame_time_s=float(eval_cfg.get("frame_time_s", 0.02)),
                hierarchical_cfg=hierarchical_cfg,
            )
            if candidates:
                w_frames = _window_frames(
                    float(hierarchical_cfg["window_len_s"]),
                    float(eval_cfg.get("frame_time_s", 0.02)),
                )
                start_frame, end_frame = candidates[0]
                sample_x_sel = {
                    key: _slice_or_pad_window(arr, start_frame, end_frame, w_frames)
                    for key, arr in sample_x_sel.items()
                }
        latency = _latency_stats(model=model, x_dict=sample_x_sel, warmup=warmup, iters=iters)

        metrics["edge"] = {
            "params": int(params),
            "checkpoint_size_mb": float(checkpoint_size_mb),
            "cpu_latency_ms": {
                "median": latency["median"],
                "p90": latency["p90"],
            },
        }
        profile_metrics[profile_name] = metrics
        all_predictions.extend(predictions)
        all_ground_truth.extend({**row, "profile": profile_name} for row in ground_truth)

    primary = selected_profiles[0]
    report_md = _render_profile_report(
        profile_metrics=profile_metrics,
        config=config,
        event_mode=event_mode,
    )

    run_dir = create_run_dir(
        base_dir=output_dir, config_dict=config, command_str="xrfv2-edge-tal event-eval"
    )
    payload = {
        "task": "phone_interaction_event",
        "event_mode": event_mode,
        "profile": primary,
        "profiles": selected_profiles,
        "event_metrics": profile_metrics[primary],
        "profile_metrics": profile_metrics,
        "labels": {
            "source_modality": source_modality,
            "positive_label_ids": sorted(int(x) for x in positive_ids),
        },
        "normalization": {
            "enabled": norm_enabled,
            "clip": norm_clip,
        },
        "hierarchical": hierarchical_cfg if event_mode == "hierarchical" else {},
    }
    write_metrics(run_dir, payload)
    (run_dir / "profile_metrics.json").write_text(
        json.dumps(profile_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "profile_report.md").write_text(report_md, encoding="utf-8")
    (run_dir / "event_predictions.json").write_text(
        json.dumps(all_predictions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "event_ground_truth.json").write_text(
        json.dumps(all_ground_truth, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "dataset_fingerprint.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")

    return run_dir


__all__ = ["eval_event_main"]
