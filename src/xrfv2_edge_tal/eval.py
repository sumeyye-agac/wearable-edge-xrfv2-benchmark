"""Evaluation entrypoint."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from xrfv2_edge_tal.artifacts import create_run_dir, write_metrics
from xrfv2_edge_tal.checkpoint import load_checkpoint
from xrfv2_edge_tal.data.adapters import DummyAdapter, XRFV2H5Adapter
from xrfv2_edge_tal.decoding import decode_argmax_probs, decode_per_class_threshold
from xrfv2_edge_tal.metrics.tal_map import (
    ap_by_class_at_tiou,
    map_over_thresholds,
    match_predictions_at_tiou,
)
from xrfv2_edge_tal.models.factory import build_model
from xrfv2_edge_tal.paper_track import (
    aggregate_window_probs,
    make_windows,
    resample_sample,
    to_frame_segments,
)
from xrfv2_edge_tal.postprocess.nms import temporal_nms


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _adapter_from_name(adapter_name: str, data_root: str, seed: int) -> DummyAdapter | XRFV2H5Adapter:
    if adapter_name == "dummy":
        return DummyAdapter(seed=seed)
    if adapter_name == "xrfv2":
        return XRFV2H5Adapter(data_root)
    raise ValueError(f"Unsupported adapter: {adapter_name}")


def _parse_modalities(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
        return items or None
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
        return items or None
    raise ValueError(f"modalities must be list[str] or comma-separated string, got: {type(value)!r}")


def _select_modalities(x: dict[str, np.ndarray], selected: list[str] | None) -> dict[str, np.ndarray]:
    if not selected:
        return x
    out = {k: v for k, v in x.items() if k in selected}
    if not out:
        available = ", ".join(sorted(x.keys()))
        wanted = ", ".join(selected)
        raise ValueError(f"Requested modalities not found. wanted=[{wanted}] available=[{available}]")
    return out


def eval_main(
    checkpoint: str,
    config: dict[str, Any],
    data_root: str,
    adapter_name: str = "dummy",
    seed: int = 42,
    output_dir: str = "runs",
) -> Path:
    _set_seed(seed)

    state, metadata = load_checkpoint(checkpoint)

    model_cfg = config.get("model", {})
    decode_cfg = config.get("decode", {})
    eval_cfg = config.get("eval", {})
    runtime_cfg = config.get("runtime", {})
    data_cfg = config.get("data", {})

    model_name = str(metadata.get("model_name", model_cfg.get("name", "tiny_tcn")))
    input_dims = dict(metadata.get("input_dims", {}))
    num_classes = int(metadata.get("num_classes", model_cfg.get("num_classes", 5)))
    hidden_dim = int(metadata.get("hidden_dim", model_cfg.get("hidden_dim", 32)))
    backend = str(metadata.get("backend", runtime_cfg.get("backend", "numpy")))
    device = str(metadata.get("device", runtime_cfg.get("device", "auto")))

    if not input_dims:
        raise ValueError("Checkpoint metadata missing input_dims")

    model = build_model(
        name=model_name,
        input_dims=input_dims,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        seed=seed,
        backend=backend,
        device=device,
        kernel_size=int(model_cfg.get("kernel_size", state.get("kernel_size", 5))),
    )
    model.load_state_dict(state)

    adapter = _adapter_from_name(adapter_name=adapter_name, data_root=data_root, seed=seed)
    split = str(eval_cfg.get("split", "test"))
    max_eval_samples = int(eval_cfg.get("max_eval_samples", 0))
    max_seq_len = int(eval_cfg.get("max_seq_len", 0))
    paper_cfg = eval_cfg.get("paper_track", {})
    paper_enabled = bool(paper_cfg.get("enabled", False))
    paper_clip_len = int(paper_cfg.get("clip_len", 2048))
    paper_stride = int(paper_cfg.get("stride", 256))
    paper_min_coverage = float(paper_cfg.get("min_segment_coverage", 0.25))
    paper_resample_to = int(paper_cfg.get("resample_to", 0))

    score_threshold = float(decode_cfg.get("score_threshold", 0.5))
    min_len = int(decode_cfg.get("min_len", 3))
    nms_tiou = float(decode_cfg.get("nms_tiou", 0.5))
    background_class = int(decode_cfg.get("background_class", 0))
    decode_mode = str(decode_cfg.get("mode", "per_class")).strip().lower()
    smooth_kernel = int(decode_cfg.get("smooth_kernel", 1))
    min_gap = int(decode_cfg.get("min_gap", 0))
    selected_modalities = _parse_modalities(data_cfg.get("modalities"))

    preds: list[dict[str, Any]] = []
    gts: list[dict[str, Any]] = []

    split_ids = adapter.split_ids(split)
    if max_eval_samples > 0:
        split_ids = split_ids[:max_eval_samples]
    for sample_id in split_ids:
        x, segments, _ = adapter.get_sample(sample_id, split)
        x = _select_modalities(x, selected=selected_modalities)
        full_seq_len = int(next(iter(x.values())).shape[0])
        segments = to_frame_segments(segments=segments, seq_len=full_seq_len)
        if max_seq_len > 0:
            full_len = int(next(iter(x.values())).shape[0])
            x = {k: v[:max_seq_len] for k, v in x.items()}
            clipped: list[dict[str, Any]] = []
            for seg in segments:
                raw_start = float(seg["start"])
                raw_end = float(seg["end"])
                if raw_start >= max_seq_len:
                    continue
                clipped.append(
                    {
                        **seg,
                        "start": float(max(0.0, raw_start)),
                        "end": float(min(float(max_seq_len), raw_end)),
                    }
                )
            segments = clipped
            full_seq_len = min(full_len, max_seq_len)

        if paper_enabled:
            if paper_resample_to > 0:
                x, segments = resample_sample(x_dict=x, segments=segments, target_len=paper_resample_to)
                full_seq_len = paper_resample_to

            windows = make_windows(
                x_dict=x,
                segments=segments,
                clip_len=paper_clip_len,
                stride=paper_stride,
                min_coverage=paper_min_coverage,
            )
            window_probs: list[np.ndarray] = []
            starts: list[int] = []
            valid_lens: list[int] = []
            for window in windows:
                window_probs.append(model.predict_proba(window["x"]))
                starts.append(int(window["start"]))
                valid_lens.append(int(window["valid_len"]))
            probs = aggregate_window_probs(
                window_probs=window_probs,
                starts=starts,
                valid_lens=valid_lens,
                full_len=full_seq_len,
            )
        else:
            probs = model.predict_proba(x)

        if decode_mode == "argmax":
            decoded = decode_argmax_probs(
                probs=probs,
                score_threshold=score_threshold,
                min_len=min_len,
                background_class=background_class,
                smooth_kernel=smooth_kernel,
                min_gap=min_gap,
            )
        elif decode_mode == "per_class":
            decoded = decode_per_class_threshold(
                probs=probs,
                score_threshold=score_threshold,
                min_len=min_len,
                background_class=background_class,
            )
        else:
            raise ValueError(f"Unsupported decode.mode='{decode_mode}'. Use 'per_class' or 'argmax'.")
        decoded = temporal_nms(decoded, tiou_threshold=nms_tiou, classwise=True)

        for seg in decoded:
            preds.append(
                {
                    "sample_id": sample_id,
                    "label": int(seg["label"]),
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "score": float(seg["score"]),
                }
            )

        for seg in segments:
            gts.append(
                {
                    "sample_id": sample_id,
                    "label": int(seg["label"]),
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                }
            )

    map_payload = map_over_thresholds(preds=preds, gts=gts)
    class_ap_050 = ap_by_class_at_tiou(preds=preds, gts=gts, tiou=0.5)
    prf_050 = match_predictions_at_tiou(preds=preds, gts=gts, tiou=0.5)

    pred_scores = [float(p.get("score", 0.0)) for p in preds]
    pred_score_stats = {
        "count": len(pred_scores),
        "mean": float(np.mean(pred_scores)) if pred_scores else 0.0,
        "std": float(np.std(pred_scores)) if pred_scores else 0.0,
        "p50": float(np.percentile(pred_scores, 50)) if pred_scores else 0.0,
        "p90": float(np.percentile(pred_scores, 90)) if pred_scores else 0.0,
        "max": float(np.max(pred_scores)) if pred_scores else 0.0,
    }

    pred_by_class = Counter(int(p["label"]) for p in preds)
    gt_by_class = Counter(int(g["label"]) for g in gts)
    class_hist = {
        "pred": {str(k): int(v) for k, v in sorted(pred_by_class.items())},
        "gt": {str(k): int(v) for k, v in sorted(gt_by_class.items())},
    }

    metrics = {
        "eval": map_payload,
        "class_ap@0.50": {str(k): float(v) for k, v in sorted(class_ap_050.items())},
        "prf@0.50": prf_050,
        "pred_score_stats": pred_score_stats,
        "class_hist": class_hist,
        "num_predictions": len(preds),
        "num_gt_segments": len(gts),
        "decode": {
            "mode": decode_mode,
            "score_threshold": score_threshold,
            "min_len": min_len,
            "nms_tiou": nms_tiou,
            "background_class": background_class,
            "smooth_kernel": smooth_kernel,
            "min_gap": min_gap,
        },
        "selected_modalities": selected_modalities or [],
        "paper_track": {
            "enabled": paper_enabled,
            "clip_len": paper_clip_len,
            "stride": paper_stride,
            "min_segment_coverage": paper_min_coverage,
            "resample_to": paper_resample_to,
        },
    }

    run_dir = create_run_dir(base_dir=output_dir, config_dict=config, command_str="xrfv2-edge-tal eval")
    write_metrics(run_dir, metrics)
    (run_dir / "dataset_fingerprint.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "eval_predictions.json").write_text(
        json.dumps(preds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "eval_ground_truth.json").write_text(
        json.dumps(gts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "eval_diagnostics.json").write_text(
        json.dumps(
            {
                "class_ap@0.50": {str(k): float(v) for k, v in sorted(class_ap_050.items())},
                "prf@0.50": prf_050,
                "pred_score_stats": pred_score_stats,
                "class_hist": class_hist,
                "num_predictions": len(preds),
                "num_gt_segments": len(gts),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return run_dir
