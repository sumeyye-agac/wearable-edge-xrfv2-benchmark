"""Training entrypoint and training loop."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from xrfv2_edge_tal.artifacts import create_run_dir, write_metrics
from xrfv2_edge_tal.checkpoint import load_checkpoint, save_checkpoint
from xrfv2_edge_tal.data.adapters import DummyAdapter, XRFV2H5Adapter
from xrfv2_edge_tal.data.prepare import compute_dataset_fingerprint
from xrfv2_edge_tal.models.factory import build_model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _adapter_from_name(adapter_name: str, data_root: str, seed: int) -> DummyAdapter | XRFV2H5Adapter:
    if adapter_name == "dummy":
        return DummyAdapter(seed=seed)
    if adapter_name == "xrfv2":
        return XRFV2H5Adapter(data_root)
    raise ValueError(f"Unsupported adapter: {adapter_name}")


def segments_to_frame_labels(segments: list[dict[str, float | int]], seq_len: int, num_classes: int) -> np.ndarray:
    labels = np.zeros((seq_len,), dtype=np.int64)
    for seg in segments:
        start = max(0, int(float(seg["start"])))
        end = min(seq_len, int(float(seg["end"])))
        label = int(seg["label"])
        label = min(max(label, 0), num_classes - 1)
        if end > start:
            labels[start:end] = label
    return labels


def _infer_input_dims(adapter: DummyAdapter | XRFV2H5Adapter, split: str) -> dict[str, int]:
    ids = adapter.split_ids(split)
    if not ids:
        raise ValueError(f"No samples found in split '{split}'")
    x, _, _ = adapter.get_sample(ids[0], split)
    return {modality: int(arr.shape[1]) for modality, arr in x.items()}


def _save_dataset_fingerprint(run_dir: Path, data_root: str, adapter: DummyAdapter | XRFV2H5Adapter) -> None:
    manifest = []
    for split in ["train", "test"]:
        for sample_id in adapter.split_ids(split):
            x, _, _ = adapter.get_sample(sample_id, split)
            manifest.append({"sample_id": sample_id, "modalities": sorted(list(x.keys()))})
    fp = compute_dataset_fingerprint(data_root=data_root, manifest=manifest)
    (run_dir / "dataset_fingerprint.json").write_text(
        json.dumps(fp, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_teacher_model(teacher_checkpoint: str, seed: int):
    state, metadata = load_checkpoint(teacher_checkpoint)
    teacher = build_model(
        name=str(metadata["model_name"]),
        input_dims=dict(metadata["input_dims"]),
        num_classes=int(metadata["num_classes"]),
        hidden_dim=int(metadata["hidden_dim"]),
        seed=seed,
        kernel_size=int(state.get("kernel_size", 5)),
    )
    teacher.load_state_dict(state)
    return teacher


def train_main(
    config: dict[str, Any],
    data_root: str,
    adapter_name: str = "dummy",
    seed: int = 42,
    runs_dir: str = "runs",
) -> Path:
    _set_seed(seed)

    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})

    epochs = int(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 1e-2))
    modality_dropout_p = float(train_cfg.get("modality_dropout_p", 0.1))
    kd_cfg = train_cfg.get("distillation", {})
    kd_enabled = bool(kd_cfg.get("enabled", False))
    kd_weight = float(kd_cfg.get("weight", 0.3))
    kd_temperature = float(kd_cfg.get("temperature", 2.0))
    kd_teacher_ckpt = str(kd_cfg.get("teacher_checkpoint", "")).strip()

    model_name = str(model_cfg.get("name", "tiny_tcn"))
    num_classes = int(model_cfg.get("num_classes", 5))
    hidden_dim = int(model_cfg.get("hidden_dim", 32))

    adapter = _adapter_from_name(adapter_name=adapter_name, data_root=data_root, seed=seed)
    input_dims = _infer_input_dims(adapter, split="train")

    model = build_model(
        name=model_name,
        input_dims=input_dims,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        seed=seed,
        kernel_size=int(model_cfg.get("kernel_size", 5)),
    )
    teacher_model = None
    if kd_enabled and kd_teacher_ckpt:
        teacher_model = _load_teacher_model(kd_teacher_ckpt, seed=seed)

    run_dir = create_run_dir(base_dir=runs_dir, config_dict=config, command_str="xrfv2-edge-tal train")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_ids = adapter.split_ids("train")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for sample_id in train_ids:
            x, segments, _ = adapter.get_sample(sample_id, "train")
            first = next(iter(x.values()))
            target = segments_to_frame_labels(segments, seq_len=first.shape[0], num_classes=num_classes)
            teacher_probs = teacher_model.predict_proba(x) if teacher_model is not None else None
            loss = model.train_step(
                x_dict=x,
                target=target,
                lr=lr,
                modality_dropout_p=modality_dropout_p,
                teacher_probs=teacher_probs,
                distill_weight=kd_weight if teacher_model is not None else 0.0,
                temperature=kd_temperature,
            )
            losses.append(loss)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "loss": mean_loss})

    metadata = {
        "model_name": model_name,
        "input_dims": input_dims,
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "adapter": adapter_name,
    }
    checkpoint_path = save_checkpoint(run_dir / "checkpoints" / "last.npz", model.state_dict(), metadata)

    metrics = {
        "train": {
            "epochs": epochs,
            "history": history,
            "final_loss": history[-1]["loss"] if history else 0.0,
        },
        "checkpoint": str(checkpoint_path),
    }
    write_metrics(run_dir, metrics)
    _save_dataset_fingerprint(run_dir, data_root=data_root, adapter=adapter)
    (run_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")

    return run_dir
