"""Training entrypoint and training loop."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from xrfv2_edge_tal.artifacts import create_run_dir, write_metrics
from xrfv2_edge_tal.checkpoint import load_checkpoint, save_checkpoint
from xrfv2_edge_tal.data.adapters import DummyAdapter, XRFV2H5Adapter
from xrfv2_edge_tal.data.prepare import compute_dataset_fingerprint
from xrfv2_edge_tal.models.factory import build_model
from xrfv2_edge_tal.paper_track import (
    augment_modalities,
    make_windows,
    resample_sample,
    to_frame_segments,
)


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
        raw_start = float(seg["start"])
        raw_end = float(seg["end"])
        # Support normalized [0, 1] timestamps (common in XRFV2 labels).
        if raw_end <= 1.0 and raw_start <= 1.0:
            raw_start *= seq_len
            raw_end *= seq_len
        start = max(0, int(raw_start))
        end = min(seq_len, int(raw_end))
        label = int(seg["label"])
        label = min(max(label, 0), num_classes - 1)
        if end > start:
            labels[start:end] = label
    return labels


def _epoch_lr(base_lr: float, epoch: int, epochs: int, schedule: str, min_lr_ratio: float) -> float:
    if schedule == "cosine" and epochs > 1:
        progress = float(epoch - 1) / float(max(epochs - 1, 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine))
    return float(base_lr)


def _infer_input_dims(adapter: DummyAdapter | XRFV2H5Adapter, split: str) -> dict[str, int]:
    ids = adapter.split_ids(split)
    if not ids:
        raise ValueError(f"No samples found in split '{split}'")
    x, _, _ = adapter.get_sample(ids[0], split)
    return {modality: int(arr.shape[1]) for modality, arr in x.items()}


def _truncate_segments(
    segments: list[dict[str, float | int]],
    full_len: int,
    max_len: int,
) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for seg in segments:
        raw_start = float(seg["start"])
        raw_end = float(seg["end"])
        if raw_end <= 1.0 and raw_start <= 1.0:
            raw_start *= full_len
            raw_end *= full_len
        if raw_start >= max_len:
            continue
        clipped = dict(seg)
        clipped["start"] = float(max(0.0, raw_start))
        clipped["end"] = float(min(float(max_len), raw_end))
        out.append(clipped)
    return out


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
        backend=str(metadata.get("backend", "numpy")),
        device=str(metadata.get("device", "auto")),
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
    runtime_cfg = config.get("runtime", {})

    epochs = int(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 1e-2))
    modality_dropout_p = float(train_cfg.get("modality_dropout_p", 0.1))
    max_train_samples = int(train_cfg.get("max_train_samples", 0))
    max_seq_len = int(train_cfg.get("max_seq_len", 0))
    lr_schedule = str(train_cfg.get("lr_schedule", "constant")).lower()
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    kd_cfg = train_cfg.get("distillation", {})
    kd_enabled = bool(kd_cfg.get("enabled", False))
    kd_weight = float(kd_cfg.get("weight", 0.3))
    kd_temperature = float(kd_cfg.get("temperature", 2.0))
    kd_teacher_ckpt = str(kd_cfg.get("teacher_checkpoint", "")).strip()
    paper_cfg = train_cfg.get("paper_track", {})
    paper_enabled = bool(paper_cfg.get("enabled", False))
    paper_clip_len = int(paper_cfg.get("clip_len", 2048))
    paper_stride = int(paper_cfg.get("stride", 256))
    paper_min_coverage = float(paper_cfg.get("min_segment_coverage", 0.25))
    paper_resample_to = int(paper_cfg.get("resample_to", 0))
    paper_max_windows = int(paper_cfg.get("max_windows_per_sample", 0))
    paper_noise_std = float(paper_cfg.get("noise_std", 0.0))
    paper_scale_jitter = float(paper_cfg.get("scale_jitter", 0.0))

    model_name = str(model_cfg.get("name", "tiny_tcn"))
    num_classes = int(model_cfg.get("num_classes", 5))
    hidden_dim = int(model_cfg.get("hidden_dim", 32))
    backend = str(runtime_cfg.get("backend", "numpy"))
    device = str(runtime_cfg.get("device", "auto"))

    adapter = _adapter_from_name(adapter_name=adapter_name, data_root=data_root, seed=seed)
    input_dims = _infer_input_dims(adapter, split="train")

    model = build_model(
        name=model_name,
        input_dims=input_dims,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        seed=seed,
        backend=backend,
        device=device,
        kernel_size=int(model_cfg.get("kernel_size", 5)),
    )
    teacher_model = None
    if kd_enabled and kd_teacher_ckpt:
        teacher_model = _load_teacher_model(kd_teacher_ckpt, seed=seed)

    run_dir = create_run_dir(base_dir=runs_dir, config_dict=config, command_str="xrfv2-edge-tal train")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_ids = adapter.split_ids("train")
    if max_train_samples > 0:
        train_ids = train_ids[:max_train_samples]
    history: list[dict[str, float | int]] = []
    total_steps = 0
    total_windows = 0
    aug_rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        losses: list[float] = []
        lr_epoch = _epoch_lr(
            base_lr=lr,
            epoch=epoch,
            epochs=epochs,
            schedule=lr_schedule,
            min_lr_ratio=min_lr_ratio,
        )
        epoch_windows = 0
        for sample_id in train_ids:
            x, segments, _ = adapter.get_sample(sample_id, "train")
            seq_len = int(next(iter(x.values())).shape[0])
            segments = to_frame_segments(segments=segments, seq_len=seq_len)
            if max_seq_len > 0:
                full_len = int(next(iter(x.values())).shape[0])
                x = {k: v[:max_seq_len] for k, v in x.items()}
                segments = _truncate_segments(segments, full_len=full_len, max_len=max_seq_len)

            train_items: list[tuple[dict[str, np.ndarray], list[dict[str, float | int]]]]
            if paper_enabled:
                if paper_resample_to > 0:
                    x, segments = resample_sample(x_dict=x, segments=segments, target_len=paper_resample_to)
                windows = make_windows(
                    x_dict=x,
                    segments=segments,
                    clip_len=paper_clip_len,
                    stride=paper_stride,
                    min_coverage=paper_min_coverage,
                )
                if paper_max_windows > 0:
                    windows = windows[:paper_max_windows]
                train_items = [(w["x"], w["segments"]) for w in windows]
            else:
                train_items = [(x, segments)]

            for x_item, seg_item in train_items:
                x_item = augment_modalities(
                    x_dict=x_item,
                    rng=aug_rng,
                    noise_std=paper_noise_std if paper_enabled else 0.0,
                    scale_jitter=paper_scale_jitter if paper_enabled else 0.0,
                )
                first = next(iter(x_item.values()))
                target = segments_to_frame_labels(seg_item, seq_len=first.shape[0], num_classes=num_classes)
                teacher_probs = teacher_model.predict_proba(x_item) if teacher_model is not None else None
                loss = model.train_step(
                    x_dict=x_item,
                    target=target,
                    lr=lr_epoch,
                    modality_dropout_p=modality_dropout_p,
                    teacher_probs=teacher_probs,
                    distill_weight=kd_weight if teacher_model is not None else 0.0,
                    temperature=kd_temperature,
                )
                losses.append(loss)
                total_steps += 1
                total_windows += 1
                epoch_windows += 1

        mean_loss = float(np.mean(losses)) if losses else 0.0
        epoch_sec = float(time.perf_counter() - t0)
        history.append(
            {
                "epoch": epoch,
                "loss": mean_loss,
                "lr": lr_epoch,
                "epoch_seconds": epoch_sec,
                "samples": len(train_ids),
                "windows": epoch_windows,
                "samples_per_sec": float(len(train_ids) / max(epoch_sec, 1e-9)),
                "windows_per_sec": float(epoch_windows / max(epoch_sec, 1e-9)),
            }
        )

    metadata = {
        "model_name": model_name,
        "input_dims": input_dims,
        "num_classes": num_classes,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "adapter": adapter_name,
        "backend": backend,
        "device": device,
    }
    checkpoint_path = save_checkpoint(run_dir / "checkpoints" / "last.npz", model.state_dict(), metadata)

    metrics = {
        "train": {
            "epochs": epochs,
            "history": history,
            "final_loss": history[-1]["loss"] if history else 0.0,
            "num_train_samples": len(train_ids),
            "total_steps": total_steps,
            "num_train_windows": total_windows,
            "lr_schedule": lr_schedule,
            "paper_track": {
                "enabled": paper_enabled,
                "clip_len": paper_clip_len,
                "stride": paper_stride,
                "min_segment_coverage": paper_min_coverage,
                "resample_to": paper_resample_to,
                "noise_std": paper_noise_std,
                "scale_jitter": paper_scale_jitter,
            },
        },
        "checkpoint": str(checkpoint_path),
    }
    write_metrics(run_dir, metrics)
    _save_dataset_fingerprint(run_dir, data_root=data_root, adapter=adapter)
    (run_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")

    return run_dir
