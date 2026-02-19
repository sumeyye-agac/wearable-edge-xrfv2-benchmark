"""XRF V2 label resolution and binary target utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_LABEL_KEYS = ["labels", "label_map", "actions", "action_map", "id2label", "label2id"]
_SEGMENT_INFO_KEYS = ["id2action", "action_map", "label_map", "id2label", "label2id"]


def _normalize_text(text: str) -> str:
    lowered = str(text).strip().lower()
    lowered = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    lowered = lowered.replace("answering", "answer")
    lowered = lowered.replace("using", "use")
    tokens = [tok for tok in lowered.split() if tok not in {"the", "a", "an"}]
    return " ".join(tokens)


def _parse_mapping(payload: Any) -> dict[int, str]:
    mapping: dict[int, str] = {}

    if isinstance(payload, list):
        if all(isinstance(item, str) for item in payload):
            return {idx: str(name) for idx, name in enumerate(payload)}
        for item in payload:
            if isinstance(item, dict):
                item_id = item.get("id", item.get("label_id"))
                item_name = item.get("name", item.get("label", item.get("action")))
                if isinstance(item_id, (int, float, np.integer, np.floating)) and isinstance(
                    item_name, str
                ):
                    mapping[int(item_id)] = item_name
        return mapping

    if not isinstance(payload, dict):
        return mapping

    numeric_keys = all(str(key).strip().lstrip("-").isdigit() for key in payload.keys())
    if numeric_keys:
        for key, value in payload.items():
            if isinstance(value, str):
                mapping[int(key)] = value
            elif isinstance(value, dict):
                maybe_name = value.get("name", value.get("label", value.get("action")))
                if isinstance(maybe_name, str):
                    mapping[int(key)] = maybe_name
        return mapping

    numeric_values = all(
        isinstance(value, (int, float, np.integer, np.floating)) for value in payload.values()
    )
    if numeric_values:
        for name, value in payload.items():
            mapping[int(value)] = str(name)
        return mapping

    # mixed dictionaries such as {"answer": {"id": 1, "name": "Answer the phone"}}
    for _, value in payload.items():
        if not isinstance(value, dict):
            continue
        item_id = value.get("id", value.get("label_id"))
        item_name = value.get("name", value.get("label", value.get("action")))
        if isinstance(item_id, (int, float, np.integer, np.floating)) and isinstance(
            item_name, str
        ):
            mapping[int(item_id)] = item_name

    return mapping


def _load_info_json(data_root: str | Path) -> dict[str, Any]:
    info_path = Path(data_root) / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json at: {info_path}")
    raw = json.loads(info_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected info.json to be a JSON object, got: {type(raw)!r}")
    return raw


def _extract_label_mapping(info: dict[str, Any]) -> tuple[dict[int, str], list[str]]:
    extracted: dict[int, str] = {}
    keys_found = [key for key in _LABEL_KEYS if key in info]
    for key in keys_found:
        extracted.update(_parse_mapping(info[key]))
    segment_info = info.get("segment_info")
    if isinstance(segment_info, dict):
        for key in _SEGMENT_INFO_KEYS:
            if key in segment_info:
                keys_found.append(f"segment_info.{key}")
                extracted.update(_parse_mapping(segment_info[key]))
    return extracted, keys_found


def resolve_positive_label_ids(
    data_root: str | Path,
    positive_action_names: list[str],
    fallback_positive_label_ids: list[int] | None = None,
) -> set[int]:
    """Resolve positive action names to label ids from info.json, with config fallback."""
    info = _load_info_json(data_root)
    extracted, keys_found = _extract_label_mapping(info)

    wanted = {_normalize_text(name) for name in positive_action_names}
    resolved = {
        int(label_id)
        for label_id, label_name in extracted.items()
        if _normalize_text(label_name) in wanted
    }

    if resolved:
        return resolved

    if fallback_positive_label_ids:
        return {int(x) for x in fallback_positive_label_ids}

    expected = ", ".join(_LABEL_KEYS)
    found = ", ".join(keys_found) if keys_found else "<none>"
    parsed = ", ".join(f"{k}:{v}" for k, v in sorted(extracted.items())) if extracted else "<none>"
    wanted_joined = ", ".join(sorted(wanted)) if wanted else "<none>"
    raise ValueError(
        "Could not resolve positive label ids from info.json. "
        f"Found keys: {found}. Expected one of: {expected}. "
        f"Parsed label map: {parsed}. Wanted action names: {wanted_joined}. "
        "Use config fallback labels.positive_label_ids=[...]."
    )


def resolve_proxy_label_ids(
    data_root: str | Path,
    keywords: list[str] | None = None,
    fallback_positive_label_ids: list[int] | None = None,
) -> set[int]:
    """Resolve a physically observable proxy label set by action-name keywords."""
    info = _load_info_json(data_root)
    extracted, keys_found = _extract_label_mapping(info)
    if not keywords:
        keywords = ["head", "face", "phone", "ear", "glasses"]
    normalized_keywords = [_normalize_text(k) for k in keywords if str(k).strip()]
    resolved = {
        int(label_id)
        for label_id, label_name in extracted.items()
        if any(keyword in _normalize_text(label_name) for keyword in normalized_keywords)
    }
    if resolved:
        return resolved
    if fallback_positive_label_ids:
        return {int(x) for x in fallback_positive_label_ids}

    found = ", ".join(keys_found) if keys_found else "<none>"
    parsed = ", ".join(f"{k}:{v}" for k, v in sorted(extracted.items())) if extracted else "<none>"
    raise ValueError(
        "Could not resolve proxy positive label ids from info.json. "
        f"Found keys: {found}. Parsed label map: {parsed}. "
        f"Keywords: {', '.join(normalized_keywords) if normalized_keywords else '<none>'}. "
        "Use config fallback labels.positive_label_ids=[...]."
    )


def build_binary_frame_labels(
    segments: list[dict[str, float | int]],
    seq_len: int,
    positive_label_ids: set[int],
) -> np.ndarray:
    """Convert multi-class segments into binary frame labels (0 background, 1 positive)."""
    labels = np.zeros((int(seq_len),), dtype=np.int64)
    if seq_len <= 0:
        return labels

    positives = {int(x) for x in positive_label_ids}
    for seg in segments:
        label = int(seg.get("label", 0))
        if label not in positives:
            continue
        raw_start = float(seg.get("start", 0.0))
        raw_end = float(seg.get("end", 0.0))
        if raw_start <= 1.0 and raw_end <= 1.0:
            raw_start *= seq_len
            raw_end *= seq_len

        start = max(0, int(raw_start))
        end = min(int(seq_len), int(raw_end))
        if end > start:
            labels[start:end] = 1
    return labels


__all__ = ["build_binary_frame_labels", "resolve_positive_label_ids", "resolve_proxy_label_ids"]
