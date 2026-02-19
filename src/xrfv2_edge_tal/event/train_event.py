"""Training entrypoint for phone-interaction event detection."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from xrfv2_edge_tal.artifacts import create_run_dir, write_metrics
from xrfv2_edge_tal.checkpoint import load_checkpoint, save_checkpoint
from xrfv2_edge_tal.data.adapters import DummyAdapter, XRFV2H5Adapter
from xrfv2_edge_tal.data.prepare import compute_dataset_fingerprint
from xrfv2_edge_tal.event.preprocess import normalization_config, normalize_modalities
from xrfv2_edge_tal.labels.xrfv2_labels import build_binary_frame_labels, resolve_positive_label_ids
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


def _resolve_profile_name(config: dict[str, Any], profile: str | None) -> str:
    data_cfg = config.get("data", {})
    default_profile = str(data_cfg.get("default_profile", "earbuds_glasses"))
    selected = profile or str(data_cfg.get("selected_profile", default_profile))
    profiles = data_cfg.get("profiles", {})
    if not isinstance(profiles, dict) or selected not in profiles:
        available = (
            ", ".join(sorted(str(k) for k in profiles.keys()))
            if isinstance(profiles, dict)
            else "<none>"
        )
        raise ValueError(
            f"Profile '{selected}' not found under data.profiles. Available: {available}"
        )
    return selected


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


def _select_modalities_by_names(
    x: dict[str, np.ndarray],
    requested_modalities: list[str],
) -> dict[str, np.ndarray]:
    raw_keys = resolve_modalities_to_raw_keys(
        available_modalities=x.keys(),
        requested_modalities=[str(item) for item in requested_modalities],
    )
    return {key: x[key] for key in raw_keys if key in x}


def _resolve_positive_ids(
    config: dict[str, Any],
    adapter_name: str,
    data_root: str,
) -> set[int]:
    labels_cfg = config.get("labels", {})
    explicit = labels_cfg.get("positive_label_ids", [])
    names = labels_cfg.get("positive_action_names", ["Answer the phone", "Use phone"])

    if explicit:
        return {int(x) for x in explicit}
    if adapter_name == "dummy":
        return {1, 2}
    return resolve_positive_label_ids(
        data_root=data_root,
        positive_action_names=[str(x) for x in names],
        fallback_positive_label_ids=None,
    )


def _resolve_label_source_modality(config: dict[str, Any]) -> str:
    labels_cfg = config.get("labels", {})
    source = str(labels_cfg.get("source_modality", "imu")).strip()
    return source or "imu"


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


def _infer_input_dims(
    adapter: DummyAdapter | XRFV2H5Adapter,
    profile_name: str,
    config: dict[str, Any],
) -> dict[str, int]:
    sample_id = adapter.split_ids("train")[0]
    x, _, _ = adapter.get_sample(sample_id, "train")
    selected = _select_modalities_for_profile(x, config=config, profile_name=profile_name)
    return {key: int(arr.shape[1]) for key, arr in selected.items()}


def _load_model_from_checkpoint(checkpoint: str, seed: int):
    state, metadata = load_checkpoint(checkpoint)
    model = build_model(
        name=str(metadata["model_name"]),
        input_dims=dict(metadata["input_dims"]),
        num_classes=int(metadata["num_classes"]),
        hidden_dim=int(metadata["hidden_dim"]),
        seed=seed,
        backend=str(metadata.get("backend", "torch")),
        device=str(metadata.get("device", "auto")),
        kernel_size=int(state.get("kernel_size", 5)),
        tcn_layers=int(state.get("tcn_layers", metadata.get("tcn_layers", 1))),
    )
    model.load_state_dict(state)
    return model, metadata


def train_event_main(
    config: dict[str, Any],
    data_root: str,
    adapter_name: str = "dummy",
    seed: int = 42,
    runs_dir: str = "runs",
    profile: str | None = None,
) -> Path:
    _set_seed(seed)

    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    runtime_cfg = config.get("runtime", {})
    data_cfg = config.get("data", {})

    epochs = int(train_cfg.get("epochs", 3))
    lr = float(train_cfg.get("lr", 1e-3))
    modality_dropout_p = float(train_cfg.get("modality_dropout_p", 0.1))
    max_train_samples = int(train_cfg.get("max_train_samples", 0))
    background_label = int(train_cfg.get("background_label", 0))
    loss_cfg = train_cfg.get("loss", {})
    focal_gamma = float(loss_cfg.get("focal_gamma", 0.0))
    background_weight = float(loss_cfg.get("background_weight", 1.0))
    class_balance = bool(loss_cfg.get("class_balance", False))
    distill_cfg = train_cfg.get("distillation", {})
    distill_enabled = bool(distill_cfg.get("enabled", False))
    distill_weight = float(distill_cfg.get("weight", 0.3))
    distill_temperature = float(distill_cfg.get("temperature", 2.0))
    teacher_checkpoint = str(distill_cfg.get("teacher_checkpoint", "")).strip()

    profile_name = _resolve_profile_name(config=config, profile=profile)
    adapter = _adapter_from_name(adapter_name=adapter_name, data_root=data_root, seed=seed)
    source_modality = _resolve_label_source_modality(config)
    norm_enabled, norm_clip = normalization_config(data_cfg)
    input_dims = _infer_input_dims(adapter=adapter, profile_name=profile_name, config=config)
    positive_ids = _resolve_positive_ids(
        config=config, adapter_name=adapter_name, data_root=data_root
    )

    model_name = str(model_cfg.get("name", "tiny_tcn"))
    hidden_dim = int(model_cfg.get("hidden_dim", 24))
    backend = str(runtime_cfg.get("backend", "torch"))
    device = str(runtime_cfg.get("device", "auto"))

    model = build_model(
        name=model_name,
        input_dims=input_dims,
        num_classes=2,
        hidden_dim=hidden_dim,
        seed=seed,
        backend=backend,
        device=device,
        kernel_size=int(model_cfg.get("kernel_size", 5)),
        tcn_layers=int(model_cfg.get("tcn_layers", 1)),
    )
    teacher_model = None
    teacher_modalities: list[str] = []
    if distill_enabled and teacher_checkpoint:
        teacher_model, teacher_meta = _load_model_from_checkpoint(teacher_checkpoint, seed=seed)
        teacher_modalities = [str(x) for x in teacher_meta.get("selected_modalities", [])]
        if not teacher_modalities:
            teacher_profile = str(distill_cfg.get("teacher_profile", ""))
            if teacher_profile and teacher_profile in config.get("data", {}).get("profiles", {}):
                teacher_modalities = [str(x) for x in config["data"]["profiles"][teacher_profile]]
            else:
                teacher_modalities = [str(x) for x in config["data"]["profiles"][profile_name]]

    run_dir = create_run_dir(
        base_dir=runs_dir, config_dict=config, command_str="xrfv2-edge-tal event-train"
    )
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_ids = adapter.split_ids("train")
    if max_train_samples > 0:
        train_ids = train_ids[:max_train_samples]

    history: list[dict[str, float | int]] = []
    total_steps = 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        losses: list[float] = []
        for sample_id in train_ids:
            x, segments_raw, _ = adapter.get_sample(sample_id, "train")
            segments = _segments_for_sample(
                adapter=adapter,
                sample_id=sample_id,
                split="train",
                fallback_segments=segments_raw,
                source_modality=source_modality,
            )
            x_sel = _select_modalities_for_profile(x, config=config, profile_name=profile_name)
            x_sel = normalize_modalities(
                x_sel,
                enabled=norm_enabled,
                clip=norm_clip,
            )
            seq_len = int(next(iter(x_sel.values())).shape[0])
            target = build_binary_frame_labels(
                segments=segments,
                seq_len=seq_len,
                positive_label_ids=positive_ids,
            )
            teacher_probs = None
            if teacher_model is not None and teacher_modalities:
                x_teacher = _select_modalities_by_names(x, teacher_modalities)
                if x_teacher:
                    x_teacher = normalize_modalities(
                        x_teacher,
                        enabled=norm_enabled,
                        clip=norm_clip,
                    )
                    teacher_probs = teacher_model.predict_proba(x_teacher)
            loss = model.train_step(
                x_dict=x_sel,
                target=target,
                lr=lr,
                modality_dropout_p=modality_dropout_p,
                teacher_probs=teacher_probs,
                distill_weight=distill_weight if teacher_probs is not None else 0.0,
                temperature=distill_temperature,
                focal_gamma=focal_gamma,
                background_label=background_label,
                background_weight=background_weight,
                class_balance=class_balance,
            )
            losses.append(float(loss))
            total_steps += 1

        epoch_s = float(time.perf_counter() - t0)
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else 0.0,
                "samples": len(train_ids),
                "epoch_seconds": epoch_s,
            }
        )

    metadata = {
        "task": "phone_interaction_event",
        "model_name": model_name,
        "input_dims": input_dims,
        "num_classes": 2,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "adapter": adapter_name,
        "backend": backend,
        "device": device,
        "tcn_layers": int(model_cfg.get("tcn_layers", 1)),
        "selected_profile": profile_name,
        "selected_modalities": config["data"]["profiles"][profile_name],
        "positive_label_ids": sorted(int(x) for x in positive_ids),
    }
    checkpoint_path = save_checkpoint(
        run_dir / "checkpoints" / "last.npz", model.state_dict(), metadata
    )

    manifest = []
    for split in ["train", "test"]:
        for sample_id in adapter.split_ids(split):
            x, _, _ = adapter.get_sample(sample_id, split)
            manifest.append({"sample_id": sample_id, "modalities": sorted(x.keys())})
    fingerprint = compute_dataset_fingerprint(data_root=data_root, manifest=manifest)

    metrics = {
        "task": "phone_interaction_event",
        "profile": profile_name,
        "train": {
            "epochs": epochs,
            "history": history,
            "final_loss": history[-1]["loss"] if history else 0.0,
            "num_train_samples": len(train_ids),
            "total_steps": total_steps,
            "loss": {
                "focal_gamma": focal_gamma,
                "background_weight": background_weight,
                "class_balance": class_balance,
            },
        },
        "labels": {
            "source_modality": source_modality,
            "positive_label_ids": sorted(int(x) for x in positive_ids),
        },
        "distillation": {
            "enabled": bool(teacher_model is not None and distill_enabled),
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_modalities": teacher_modalities,
            "weight": distill_weight,
            "temperature": distill_temperature,
        },
        "normalization": {
            "enabled": norm_enabled,
            "clip": norm_clip,
        },
        "checkpoint": str(checkpoint_path),
    }
    write_metrics(run_dir, metrics)
    (run_dir / "dataset_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "profile_metrics.json").write_text("{}\n", encoding="utf-8")

    return run_dir


__all__ = ["train_event_main"]
