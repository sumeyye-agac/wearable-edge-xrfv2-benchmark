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
from xrfv2_edge_tal.event.candidates import default_candidate_config
from xrfv2_edge_tal.event.preprocess import normalization_config, normalize_modalities
from xrfv2_edge_tal.event.window_dataset import build_windows_for_sample
from xrfv2_edge_tal.labels.xrfv2_labels import (
    build_binary_frame_labels,
    resolve_positive_label_ids,
    resolve_proxy_label_ids,
)
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
    train_cfg = config.get("train", {})
    eval_cfg = config.get("eval", {})
    mode = str(train_cfg.get("event_mode", eval_cfg.get("event_mode", "flat"))).strip().lower()
    if mode not in {"flat", "hierarchical"}:
        raise ValueError(f"Unsupported train.event_mode={mode}. Expected flat|hierarchical.")
    return mode


def _resolve_hierarchical_cfg(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = config.get("train", {})
    raw = train_cfg.get("hierarchical", {})
    if not isinstance(raw, dict):
        raw = {}
    cfg = dict(default_candidate_config(raw))
    supervision = str(raw.get("window_label_mode", "pooled_max")).strip().lower()
    if supervision not in {"frame_dense", "pooled_max", "pooled_mean"}:
        raise ValueError(
            f"Unsupported train.hierarchical.window_label_mode={supervision}. "
            "Expected: frame_dense|pooled_max|pooled_mean."
        )
    cfg["window_label_mode"] = supervision
    cfg["positive_repeat"] = max(1, int(raw.get("positive_repeat", 1)))
    max_negative_ratio = float(raw.get("max_negative_ratio", 0.0))
    cfg["max_negative_ratio"] = max(0.0, max_negative_ratio)
    return cfg


def _resolve_hard_negative_cfg(config: dict[str, Any]) -> dict[str, Any]:
    train_cfg = config.get("train", {})
    raw = train_cfg.get("hard_negative_mining", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "start_epoch": max(1, int(raw.get("start_epoch", 1))),
        "subset_samples": max(0, int(raw.get("subset_samples", 256))),
        "score_threshold": float(raw.get("score_threshold", 0.6)),
        "weight": float(raw.get("weight", 2.0)),
        "max_windows_per_sample": max(1, int(raw.get("max_windows_per_sample", 2))),
    }


def _rebalance_windows(
    windows: list[dict[str, Any]],
    *,
    max_negative_ratio: float,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], int, int, int]:
    if max_negative_ratio <= 0.0:
        positives = int(sum(int(w.get("y", 0)) == 1 for w in windows))
        negatives = int(len(windows) - positives)
        return windows, positives, negatives, negatives

    positives = [w for w in windows if int(w.get("y", 0)) == 1]
    negatives = [w for w in windows if int(w.get("y", 0)) != 1]
    original_negatives = len(negatives)
    if positives and negatives:
        neg_cap = max(1, int(np.floor(float(max_negative_ratio) * len(positives))))
        if len(negatives) > neg_cap:
            order = np.arange(len(negatives))
            rng.shuffle(order)
            negatives = [negatives[int(i)] for i in order[:neg_cap]]
    merged = list(positives) + list(negatives)
    if merged:
        order = np.arange(len(merged))
        rng.shuffle(order)
        merged = [merged[int(i)] for i in order]
    return merged, len(positives), original_negatives, len(negatives)


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


def _mine_hard_negative_windows(
    *,
    model: Any,
    adapter: DummyAdapter | XRFV2H5Adapter,
    sample_ids: list[str],
    config: dict[str, Any],
    profile_name: str,
    source_modality: str,
    positive_ids: set[int],
    frame_time_s: float,
    window_len_s: float,
    hierarchical_cfg: dict[str, Any],
    hard_negative_cfg: dict[str, Any],
    norm_enabled: bool,
    norm_clip: float,
) -> dict[str, list[dict[str, Any]]]:
    if not bool(hard_negative_cfg["enabled"]):
        return {}

    subset = int(hard_negative_cfg["subset_samples"])
    mining_ids = sample_ids[:subset] if subset > 0 else list(sample_ids)

    candidate_cfg = dict(hierarchical_cfg)
    candidate_cfg["include_gt_windows"] = False
    candidate_cfg["max_windows"] = int(hard_negative_cfg["max_windows_per_sample"])
    score_threshold = float(hard_negative_cfg["score_threshold"])
    weight = max(1.0, float(hard_negative_cfg["weight"]))

    hard_by_sample: dict[str, list[dict[str, Any]]] = {}
    for sample_id in mining_ids:
        x, segments_raw, _ = adapter.get_sample(sample_id, "train")
        segments = _segments_for_sample(
            adapter=adapter,
            sample_id=sample_id,
            split="train",
            fallback_segments=segments_raw,
            source_modality=source_modality,
        )
        x_sel = _select_modalities_for_profile(x, config=config, profile_name=profile_name)
        x_sel = normalize_modalities(x_sel, enabled=norm_enabled, clip=norm_clip)
        windows = build_windows_for_sample(
            x_dict=x_sel,
            segments=segments,
            positive_ids=positive_ids,
            profile_modalities=sorted(list(x_sel.keys())),
            frame_time_s=frame_time_s,
            candidate_cfg=candidate_cfg,
            window_len_s=window_len_s,
            sample_id=sample_id,
        )

        mined: list[dict[str, Any]] = []
        for window in windows:
            if int(window["y"]) != 0:
                continue
            x_win = normalize_modalities(
                dict(window["x_window"]),
                enabled=norm_enabled,
                clip=norm_clip,
            )
            score = float(np.mean(model.predict_proba(x_win)[:, 1]))
            if score < score_threshold:
                continue
            mined_window = dict(window)
            mined_window["weight"] = weight
            mined_window["score"] = score
            mined.append(mined_window)
        if mined:
            hard_by_sample[sample_id] = mined[: int(hard_negative_cfg["max_windows_per_sample"])]

    return hard_by_sample


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
    event_mode = _resolve_event_mode(config)
    hierarchical_cfg = _resolve_hierarchical_cfg(config)
    frame_time_s = float(config.get("eval", {}).get("frame_time_s", 0.02))
    window_len_s = float(hierarchical_cfg.get("window_len_s", 1.5))

    epochs = int(train_cfg.get("epochs", 3))
    lr = float(train_cfg.get("lr", 1e-3))
    modality_dropout_p = float(train_cfg.get("modality_dropout_p", 0.1))
    max_train_samples = int(train_cfg.get("max_train_samples", 0))
    background_label = int(train_cfg.get("background_label", 0))
    loss_cfg = train_cfg.get("loss", {})
    focal_gamma = float(loss_cfg.get("focal_gamma", 0.0))
    background_weight = float(loss_cfg.get("background_weight", 1.0))
    class_balance = bool(loss_cfg.get("class_balance", train_cfg.get("class_balance", False)))
    hard_negative_cfg = _resolve_hard_negative_cfg(config)
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
    total_windows = 0
    total_positive_windows = 0
    total_negative_windows = 0
    total_negative_windows_kept = 0
    total_positive_weighted_steps = 0
    hard_negative_pool: dict[str, list[dict[str, Any]]] = {}
    hard_negative_windows_last = 0
    hard_negative_weighted_steps = 0
    hard_negative_history: list[dict[str, float | int]] = []
    rng = np.random.default_rng(seed)
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
            x_sel = normalize_modalities(x_sel, enabled=norm_enabled, clip=norm_clip)

            if event_mode == "flat":
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
                continue

            windows = build_windows_for_sample(
                x_dict=x_sel,
                segments=segments,
                positive_ids=positive_ids,
                profile_modalities=sorted(list(x_sel.keys())),
                frame_time_s=frame_time_s,
                candidate_cfg=hierarchical_cfg,
                window_len_s=window_len_s,
                sample_id=sample_id,
            )
            extra_hard = hard_negative_pool.get(sample_id, [])
            if extra_hard:
                windows = list(windows) + [dict(w) for w in extra_hard]
            windows, num_pos, num_neg_before, num_neg_after = _rebalance_windows(
                windows,
                max_negative_ratio=float(hierarchical_cfg.get("max_negative_ratio", 0.0)),
                rng=rng,
            )
            total_windows += len(windows)
            total_positive_windows += int(num_pos)
            total_negative_windows += int(num_neg_before)
            total_negative_windows_kept += int(num_neg_after)
            for window in windows:
                x_win = normalize_modalities(
                    dict(window["x_window"]),
                    enabled=norm_enabled,
                    clip=norm_clip,
                )
                win_len = int(next(iter(x_win.values())).shape[0])
                label = int(window["y"])
                label_mode = str(hierarchical_cfg.get("window_label_mode", "pooled_max"))
                if label_mode == "frame_dense":
                    target = np.full((win_len,), label, dtype=np.int64)
                    window_pooling = None
                elif label_mode == "pooled_mean":
                    target = np.asarray([label], dtype=np.int64)
                    window_pooling = "mean"
                else:
                    target = np.asarray([label], dtype=np.int64)
                    window_pooling = "max"

                teacher_probs = None
                if teacher_model is not None and teacher_modalities:
                    x_teacher_full = _select_modalities_by_names(x, teacher_modalities)
                    if x_teacher_full:
                        x_teacher_window = {
                            key: _slice_or_pad_window(
                                np.asarray(arr, dtype=np.float32),
                                start=int(window["start_frame"]),
                                end=int(window["end_frame"]),
                                target_len=win_len,
                            )
                            for key, arr in x_teacher_full.items()
                        }
                        x_teacher_window = normalize_modalities(
                            x_teacher_window,
                            enabled=norm_enabled,
                            clip=norm_clip,
                        )
                        teacher_probs = teacher_model.predict_proba(x_teacher_window)

                repeat = max(1, int(round(float(window.get("weight", 1.0)))))
                if label == 1:
                    repeat *= max(1, int(hierarchical_cfg.get("positive_repeat", 1)))
                    total_positive_weighted_steps += repeat
                if float(window.get("weight", 1.0)) > 1.0:
                    hard_negative_weighted_steps += repeat
                for _ in range(repeat):
                    loss = model.train_step(
                        x_dict=x_win,
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
                        window_pooling=window_pooling,
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
        if (
            event_mode == "hierarchical"
            and bool(hard_negative_cfg["enabled"])
            and epoch >= int(hard_negative_cfg["start_epoch"])
            and epoch < epochs
        ):
            mine_t0 = time.perf_counter()
            hard_negative_pool = _mine_hard_negative_windows(
                model=model,
                adapter=adapter,
                sample_ids=train_ids,
                config=config,
                profile_name=profile_name,
                source_modality=source_modality,
                positive_ids=positive_ids,
                frame_time_s=frame_time_s,
                window_len_s=window_len_s,
                hierarchical_cfg=hierarchical_cfg,
                hard_negative_cfg=hard_negative_cfg,
                norm_enabled=norm_enabled,
                norm_clip=norm_clip,
            )
            hard_negative_windows_last = int(sum(len(v) for v in hard_negative_pool.values()))
            hard_negative_history.append(
                {
                    "epoch": epoch,
                    "windows": hard_negative_windows_last,
                    "seconds": float(time.perf_counter() - mine_t0),
                }
            )

    metadata = {
        "task": "phone_interaction_event",
        "event_mode": event_mode,
        "model_name": model_name,
        "input_dims": input_dims,
        "num_classes": 2,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "adapter": adapter_name,
        "backend": backend,
        "device": device,
        "kernel_size": int(model_cfg.get("kernel_size", 5)),
        "tcn_layers": int(model_cfg.get("tcn_layers", 1)),
        "selected_profile": profile_name,
        "selected_modalities": config["data"]["profiles"][profile_name],
        "positive_label_ids": sorted(int(x) for x in positive_ids),
    }
    checkpoint_path = save_checkpoint(
        run_dir / "checkpoints" / "last.npz", model.state_dict(), metadata
    )

    manifest = []
    adapter_modalities = sorted(list(adapter.modalities))
    for split in ["train", "test"]:
        for sample_id in adapter.split_ids(split):
            manifest.append(
                {
                    "sample_id": sample_id,
                    "source_split": split,
                    "modalities": adapter_modalities,
                }
            )
    fingerprint = compute_dataset_fingerprint(data_root=data_root, manifest=manifest)

    metrics = {
        "task": "phone_interaction_event",
        "event_mode": event_mode,
        "profile": profile_name,
        "train": {
            "epochs": epochs,
            "history": history,
            "final_loss": history[-1]["loss"] if history else 0.0,
            "num_train_samples": len(train_ids),
            "total_steps": total_steps,
            "hierarchical": {
                "window_len_s": window_len_s,
                "candidate_cfg": hierarchical_cfg,
                "window_label_mode": str(hierarchical_cfg.get("window_label_mode", "pooled_max")),
                "positive_repeat": int(hierarchical_cfg.get("positive_repeat", 1)),
                "max_negative_ratio": float(hierarchical_cfg.get("max_negative_ratio", 0.0)),
                "num_windows": int(total_windows),
                "num_positive_windows": int(total_positive_windows),
                "num_negative_windows": int(total_negative_windows),
                "num_negative_windows_kept": int(total_negative_windows_kept),
                "num_positive_weighted_steps": int(total_positive_weighted_steps),
                "hard_negative_mining": {
                    "enabled": bool(hard_negative_cfg["enabled"]),
                    "config": hard_negative_cfg,
                    "windows_last_epoch": int(hard_negative_windows_last),
                    "weighted_steps": int(hard_negative_weighted_steps),
                    "history": hard_negative_history,
                },
            }
            if event_mode == "hierarchical"
            else {},
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
