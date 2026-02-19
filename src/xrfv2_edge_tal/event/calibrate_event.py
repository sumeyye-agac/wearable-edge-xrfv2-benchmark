"""Threshold/cooldown calibration for event detection profiles."""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from xrfv2_edge_tal.artifacts import create_run_dir, write_metrics
from xrfv2_edge_tal.checkpoint import load_checkpoint
from xrfv2_edge_tal.event.eval_event import (
    _adapter_from_name,
    _hierarchical_candidates,
    _resolve_event_mode,
    _resolve_hierarchical_cfg,
    _resolve_label_source_modality,
    _resolve_positive_ids,
    _resolve_profile_names,
    _segment_end_seconds,
    _segment_start_seconds,
    _segments_for_sample,
    _select_modalities_for_profile,
    _sequence_duration_seconds,
    _slice_or_pad_window,
    _window_frames,
)
from xrfv2_edge_tal.event.metrics import compute_event_metrics
from xrfv2_edge_tal.event.preprocess import normalization_config, normalize_modalities
from xrfv2_edge_tal.event.trigger import filter_trigger_candidates, frame_probs_to_event_triggers
from xrfv2_edge_tal.models.factory import build_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _default_thresholds() -> list[float]:
    return [round(float(x), 2) for x in np.arange(0.1, 0.95, 0.05)]


def _default_cooldowns(base: float) -> list[float]:
    vals = sorted({max(0.0, base * 0.5), base, base * 1.5})
    return [round(float(v), 3) for v in vals]


def _pick_best_row(
    rows: list[dict[str, Any]],
    metric_mode: str,
    fp_hour_budget: float | None,
) -> tuple[dict[str, Any], bool]:
    if not rows:
        raise ValueError("No calibration rows generated")

    if fp_hour_budget is not None:
        feasible = [
            r for r in rows if float(r[metric_mode]["fp_per_hour"]) <= float(fp_hour_budget)
        ]
        if feasible:
            best = max(
                feasible,
                key=lambda r: (float(r[metric_mode]["f1"]), -float(r[metric_mode]["fp_per_hour"])),
            )
            return best, True

    best = max(
        rows,
        key=lambda r: (float(r[metric_mode]["f1"]), -float(r[metric_mode]["fp_per_hour"])),
    )
    return best, False


def _render_calibration_report(
    best_by_profile: OrderedDict[str, dict[str, Any]],
    metric_mode: str,
    fp_hour_budget: float | None,
    event_mode: str,
) -> str:
    lines = [
        "# Event Calibration Report",
        "",
        f"- Event mode: `{event_mode}`",
        f"- Metric mode: `{metric_mode}`",
        f"- FP/hour budget: `{fp_hour_budget}`"
        if fp_hour_budget is not None
        else "- FP/hour budget: <none>",
        "",
        "| Profile | Best threshold | Best cooldown (s) | Onset F1 | Within F1 | Onset FP/hour | Within FP/hour | Budget met |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile, payload in best_by_profile.items():
        row = payload["best_row"]
        lines.append(
            "| "
            f"`{profile}` | {float(row['threshold']):.2f} | {float(row['cooldown_s']):.3f} | "
            f"{float(row['onset_strict']['f1']):.4f} | {float(row['within_segment']['f1']):.4f} | "
            f"{float(row['onset_strict']['fp_per_hour']):.3f} | {float(row['within_segment']['fp_per_hour']):.3f} | "
            f"{str(bool(payload['budget_met']))} |"
        )
    lines.append("")
    return "\n".join(lines)


def calibrate_event_main(
    checkpoint: str,
    config: dict[str, Any],
    data_root: str,
    adapter_name: str = "dummy",
    seed: int = 42,
    output_dir: str = "runs",
    profile: str | None = None,
    profiles: list[str] | None = None,
    thresholds: list[float] | None = None,
    cooldowns: list[float] | None = None,
    metric_mode: str = "within_segment",
    fp_hour_budget: float | None = None,
) -> Path:
    _set_seed(seed)

    if metric_mode not in {"onset_strict", "within_segment"}:
        raise ValueError("metric_mode must be one of: onset_strict, within_segment")

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
    selected_profiles = _resolve_profile_names(config=config, profile=profile, profiles=profiles)

    frame_time_s = float(eval_cfg.get("frame_time_s", 0.02))
    onset_tolerance_s = float(eval_cfg.get("onset_tolerance_s", 0.5))
    max_eval_samples = int(eval_cfg.get("max_eval_samples", 0))
    trigger_cfg = eval_cfg.get("trigger", {})
    smoothing_window = int(trigger_cfg.get("smoothing_window", 5))
    hysteresis = bool(trigger_cfg.get("hysteresis", False))
    default_threshold_off = float(trigger_cfg.get("threshold_off", 0.4))
    base_cooldown = float(trigger_cfg.get("cooldown_s", 0.75))
    min_active_s = float(trigger_cfg.get("min_active_s", 0.0))

    thresholds = [float(x) for x in (thresholds if thresholds else _default_thresholds())]
    cooldowns = [float(x) for x in (cooldowns if cooldowns else _default_cooldowns(base_cooldown))]
    norm_enabled, norm_clip = normalization_config(data_cfg)

    calibration_rows: list[dict[str, Any]] = []
    best_by_profile: OrderedDict[str, dict[str, Any]] = OrderedDict()

    split = str(eval_cfg.get("split", "test"))
    sample_ids = adapter.split_ids(split)
    if max_eval_samples > 0:
        sample_ids = sample_ids[:max_eval_samples]

    for profile_name in selected_profiles:
        cached_streams: list[tuple[str, np.ndarray | list[dict[str, float | int]], int]] = []
        duration_s = 0.0
        ground_truth_flat: list[dict[str, Any]] = []

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

            gt_segments: list[dict[str, Any]] = []
            for seg in segments:
                if int(seg["label"]) not in positive_ids:
                    continue
                row = {
                    "sample_id": sample_id,
                    "start": _segment_start_seconds(
                        seg, seq_len=seq_len, frame_time_s=frame_time_s
                    ),
                    "end": _segment_end_seconds(seg, seq_len=seq_len, frame_time_s=frame_time_s),
                }
                gt_segments.append(row)
                ground_truth_flat.append(row)

            if event_mode == "flat":
                probs = model.predict_proba(x_sel)
                if probs.ndim != 2 or probs.shape[1] < 2:
                    raise ValueError(f"Expected model output [T,2+], got {probs.shape}")
                cached_streams.append((sample_id, probs[:, 1], seq_len))
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
                        raise ValueError(f"Expected model output [W,2+], got {probs.shape}")
                    scored.append(
                        {
                            "time": float(start_frame * frame_time_s),
                            "score": float(np.mean(probs[:, 1])),
                            "frame": int(start_frame),
                        }
                    )
                cached_streams.append((sample_id, scored, seq_len))
            duration_s += _sequence_duration_seconds(
                meta=meta, seq_len=seq_len, frame_time_s=frame_time_s
            )

        profile_rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            for cooldown_s in cooldowns:
                if hysteresis:
                    threshold_off = min(default_threshold_off, float(threshold))
                else:
                    threshold_off = None

                predictions: list[dict[str, Any]] = []
                for sample_id, stream, _ in cached_streams:
                    if event_mode == "flat":
                        assert isinstance(stream, np.ndarray)
                        triggers = frame_probs_to_event_triggers(
                            probs=stream,
                            frame_time_s=frame_time_s,
                            threshold=float(threshold),
                            smoothing_window=smoothing_window,
                            cooldown_s=float(cooldown_s),
                            hysteresis=hysteresis,
                            threshold_off=threshold_off,
                            min_active_s=min_active_s,
                        )
                    else:
                        assert isinstance(stream, list)
                        triggers = filter_trigger_candidates(
                            candidates=stream,
                            threshold=float(threshold),
                            cooldown_s=float(cooldown_s),
                            hysteresis=hysteresis,
                            threshold_off=threshold_off,
                        )
                    for trigger in triggers:
                        predictions.append(
                            {
                                "sample_id": sample_id,
                                "time": float(trigger["time"]),
                                "score": float(trigger["score"]),
                            }
                        )

                metrics = compute_event_metrics(
                    predictions=predictions,
                    ground_truth=ground_truth_flat,
                    duration_s=duration_s,
                    onset_tolerance_s=onset_tolerance_s,
                )
                row = {
                    "profile": profile_name,
                    "threshold": float(threshold),
                    "cooldown_s": float(cooldown_s),
                    "num_predictions": len(predictions),
                    "num_ground_truth": len(ground_truth_flat),
                    "onset_strict": metrics["onset_strict"],
                    "within_segment": metrics["within_segment"],
                }
                profile_rows.append(row)
                calibration_rows.append(row)

        best_row, budget_met = _pick_best_row(
            rows=profile_rows,
            metric_mode=metric_mode,
            fp_hour_budget=fp_hour_budget,
        )
        best_by_profile[profile_name] = {
            "best_row": best_row,
            "budget_met": budget_met,
        }

    run_dir = create_run_dir(
        base_dir=output_dir,
        config_dict=config,
        command_str="xrfv2-edge-tal event-calibrate",
    )
    report_md = _render_calibration_report(
        best_by_profile=best_by_profile,
        metric_mode=metric_mode,
        fp_hour_budget=fp_hour_budget,
        event_mode=event_mode,
    )

    payload = {
        "task": "phone_interaction_event",
        "event_mode": event_mode,
        "calibration": {
            "metric_mode": metric_mode,
            "fp_hour_budget": fp_hour_budget,
            "thresholds": thresholds,
            "cooldowns": cooldowns,
            "profiles": selected_profiles,
            "best_by_profile": best_by_profile,
            "source_modality": source_modality,
            "hierarchical": hierarchical_cfg if event_mode == "hierarchical" else {},
        },
    }
    write_metrics(run_dir, payload)
    (run_dir / "calibration_grid.json").write_text(
        json.dumps(calibration_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "calibration_report.md").write_text(report_md, encoding="utf-8")
    (run_dir / "dataset_fingerprint.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")

    return run_dir


__all__ = ["calibrate_event_main"]
