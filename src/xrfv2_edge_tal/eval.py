"""Evaluation entrypoint."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from xrfv2_edge_tal.artifacts import create_run_dir, write_metrics
from xrfv2_edge_tal.checkpoint import load_checkpoint
from xrfv2_edge_tal.data.adapters import DummyAdapter, XRFV2H5Adapter
from xrfv2_edge_tal.decoding import decode_framewise_probs
from xrfv2_edge_tal.metrics.tal_map import map_over_thresholds
from xrfv2_edge_tal.models.factory import build_model
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

    model_name = str(metadata.get("model_name", model_cfg.get("name", "tiny_tcn")))
    input_dims = dict(metadata.get("input_dims", {}))
    num_classes = int(metadata.get("num_classes", model_cfg.get("num_classes", 5)))
    hidden_dim = int(metadata.get("hidden_dim", model_cfg.get("hidden_dim", 32)))

    if not input_dims:
        raise ValueError("Checkpoint metadata missing input_dims")

    model = build_model(
        name=model_name,
        input_dims=input_dims,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        seed=seed,
        kernel_size=int(model_cfg.get("kernel_size", state.get("kernel_size", 5))),
    )
    model.load_state_dict(state)

    adapter = _adapter_from_name(adapter_name=adapter_name, data_root=data_root, seed=seed)
    split = str(eval_cfg.get("split", "test"))

    score_threshold = float(decode_cfg.get("score_threshold", 0.5))
    min_len = int(decode_cfg.get("min_len", 3))
    nms_tiou = float(decode_cfg.get("nms_tiou", 0.5))

    preds: list[dict[str, Any]] = []
    gts: list[dict[str, Any]] = []

    for sample_id in adapter.split_ids(split):
        x, segments, _ = adapter.get_sample(sample_id, split)
        probs = model.predict_proba(x)

        decoded = decode_framewise_probs(
            probs=probs,
            score_threshold=score_threshold,
            min_len=min_len,
            background_class=0,
        )
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
    metrics = {
        "eval": map_payload,
        "num_predictions": len(preds),
        "num_gt_segments": len(gts),
    }

    run_dir = create_run_dir(base_dir=output_dir, config_dict=config, command_str="xrfv2-edge-tal eval")
    write_metrics(run_dir, metrics)
    (run_dir / "dataset_fingerprint.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "eval_predictions.json").write_text(
        json.dumps(preds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return run_dir
