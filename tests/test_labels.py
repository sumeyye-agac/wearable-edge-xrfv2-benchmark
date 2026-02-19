from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xrfv2_edge_tal.labels.xrfv2_labels import (
    build_binary_frame_labels,
    resolve_positive_label_ids,
    resolve_proxy_label_ids,
)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"labels": ["Background", "Answer the phone", "Use phone"]}, {1, 2}),
        ({"label_map": {"0": "Background", "1": "Answer the phone", "2": "Use phone"}}, {1, 2}),
        ({"actions": {"Background": 0, "Answer the phone": 7, "Use phone": 9}}, {7, 9}),
        (
            {
                "segment_info": {
                    "id2action": {
                        "0": "Background",
                        "16": "Answering Phone",
                        "21": "Using Phone",
                    }
                }
            },
            {16, 21},
        ),
        (
            {
                "id2label": {
                    "0": {"name": "Background"},
                    "3": {"name": "Answer the phone"},
                    "4": {"name": "Use phone"},
                }
            },
            {3, 4},
        ),
    ],
)
def test_resolve_positive_label_ids_variants(
    tmp_path: Path, payload: dict, expected: set[int]
) -> None:
    (tmp_path / "info.json").write_text(json.dumps(payload), encoding="utf-8")
    ids = resolve_positive_label_ids(
        data_root=tmp_path,
        positive_action_names=["Answer the phone", "Use phone"],
    )
    assert ids == expected


def test_resolve_positive_label_ids_fallback(tmp_path: Path) -> None:
    (tmp_path / "info.json").write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    ids = resolve_positive_label_ids(
        data_root=tmp_path,
        positive_action_names=["Answer the phone", "Use phone"],
        fallback_positive_label_ids=[10, 11],
    )
    assert ids == {10, 11}


def test_resolve_positive_label_ids_error_is_actionable(tmp_path: Path) -> None:
    (tmp_path / "info.json").write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    with pytest.raises(ValueError, match="labels.positive_label_ids"):
        resolve_positive_label_ids(
            data_root=tmp_path,
            positive_action_names=["Answer the phone", "Use phone"],
            fallback_positive_label_ids=None,
        )


def test_build_binary_frame_labels() -> None:
    segments = [
        {"start": 0, "end": 3, "label": 1},
        {"start": 5, "end": 7, "label": 4},
        {"start": 8, "end": 10, "label": 2},
    ]
    y = build_binary_frame_labels(segments=segments, seq_len=12, positive_label_ids={2, 4})
    assert y.dtype == np.int64
    assert y.shape == (12,)
    assert np.all(y[:5] == 0)
    assert np.all(y[5:7] == 1)
    assert np.all(y[8:10] == 1)


def test_build_binary_frame_labels_normalized_segments() -> None:
    segments = [{"start": 0.2, "end": 0.5, "label": 3}]
    y = build_binary_frame_labels(segments=segments, seq_len=10, positive_label_ids={3})
    assert np.all(y[:2] == 0)
    assert np.all(y[2:5] == 1)
    assert np.all(y[5:] == 0)


def test_resolve_proxy_label_ids_keywords(tmp_path: Path) -> None:
    payload = {
        "segment_info": {
            "id2action": {
                "3": "Drinking Water",
                "9": "Touching Face",
                "16": "Answering Phone",
                "21": "Using Phone",
            }
        }
    }
    (tmp_path / "info.json").write_text(json.dumps(payload), encoding="utf-8")
    ids = resolve_proxy_label_ids(data_root=tmp_path, keywords=["face", "phone"])
    assert ids == {9, 16, 21}


def test_resolve_proxy_label_ids_fallback(tmp_path: Path) -> None:
    (tmp_path / "info.json").write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    ids = resolve_proxy_label_ids(
        data_root=tmp_path,
        keywords=["face", "phone"],
        fallback_positive_label_ids=[5, 7],
    )
    assert ids == {5, 7}
