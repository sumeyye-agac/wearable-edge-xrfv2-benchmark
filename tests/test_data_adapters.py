from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from xrfv2_edge_tal.data.adapters import DummyAdapter, XRFV2H5Adapter
from xrfv2_edge_tal.data.prepare import prepare_dataset


def _write_tiny_xrfv2_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    with h5py.File(root / "train_data.h5", "w") as f:
        f.create_dataset("imu_phone", data=np.random.randn(2, 12, 3).astype(np.float32))
        f.create_dataset("imu_watch", data=np.random.randn(2, 12, 3).astype(np.float32))

    with h5py.File(root / "test_data.h5", "w") as f:
        g = f.create_group("modalities")
        g.create_dataset("imu_phone", data=np.random.randn(2, 10, 3).astype(np.float32))
        g.create_dataset("imu_watch", data=np.random.randn(2, 10, 3).astype(np.float32))

    train_label = {
        "0": [[1, 4, 1], [6, 8, 2]],
        "1": [[0, 3, 0.7, 3]],
    }
    test_label = {
        "0": {"imu_phone": [[2, 5, 1]], "imu_watch": [[3, 6, 1]]},
        "1": [],
    }
    info = {"modalities": ["imu_phone", "imu_watch"]}

    (root / "train_label.json").write_text(json.dumps(train_label), encoding="utf-8")
    (root / "test_label.json").write_text(json.dumps(test_label), encoding="utf-8")
    (root / "info.json").write_text(json.dumps(info), encoding="utf-8")


def test_dummy_prepare_end_to_end(tmp_path: Path) -> None:
    adapter = DummyAdapter(seed=7, num_train=4, num_test=2)
    out = prepare_dataset(
        adapter=adapter,
        data_root=tmp_path,
        output_dir=tmp_path / "processed",
        seed=7,
    )

    assert out["manifest"].exists()
    assert out["default_split"].exists()
    assert out["fingerprint"].exists()

    lines = out["manifest"].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 6


def test_xrfv2_adapter_parses_files(tmp_path: Path) -> None:
    root = tmp_path / "xrfv2"
    _write_tiny_xrfv2_dir(root)

    adapter = XRFV2H5Adapter(root)
    assert adapter.modalities == ["imu_phone", "imu_watch"]
    assert len(adapter.split_ids("train")) == 2

    x, segments, meta = adapter.get_sample("1", "train")
    assert sorted(list(x.keys())) == ["imu_phone", "imu_watch"]
    assert x["imu_phone"].shape == (12, 3)
    assert len(segments) == 1
    assert int(segments[0]["label"]) == 3
    assert meta["source"] == "train"

    x_test, segments_test, _ = adapter.get_sample("0", "test")
    assert x_test["imu_phone"].shape == (10, 3)
    assert len(segments_test) == 2


def test_xrfv2_adapter_missing_files(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    root.mkdir(parents=True)
    (root / "info.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        XRFV2H5Adapter(root)
