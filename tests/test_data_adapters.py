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


def _write_receiver_split_xrfv2_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    imu_train = np.arange(2 * 12 * 5 * 6, dtype=np.float32).reshape(2, 12, 5, 6)
    airpods_train = np.arange(2 * 12 * 9, dtype=np.float32).reshape(2, 12, 9)
    wifi_train = np.arange(2 * 12 * 3 * 3 * 30, dtype=np.float32).reshape(2, 12, 3, 3, 30)

    with h5py.File(root / "train_data.h5", "w") as f:
        f.create_dataset("imu", data=imu_train)
        f.create_dataset("airpods", data=airpods_train)
        f.create_dataset("wifi", data=wifi_train)

    with h5py.File(root / "test_data.h5", "w") as f:
        f.create_dataset("imu", data=imu_train)
        f.create_dataset("airpods", data=airpods_train)
        f.create_dataset("wifi", data=wifi_train)

    label = {"imu": {"0": [[0, 3, 1]], "1": [[5, 7, 2]]}}
    info = {"modality_list": ["imu", "wifi", "airpods"]}
    (root / "train_label.json").write_text(json.dumps(label), encoding="utf-8")
    (root / "test_label.json").write_text(json.dumps(label), encoding="utf-8")
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


def test_xrfv2_adapter_splits_imu_receivers_and_airpods_channels(tmp_path: Path) -> None:
    root = tmp_path / "xrfv2"
    _write_receiver_split_xrfv2_dir(root)
    adapter = XRFV2H5Adapter(root)

    assert "imu_gl" in adapter.modalities
    assert "imu_rp" in adapter.modalities

    x, _, _ = adapter.get_sample("0", "train")
    assert x["imu"].shape == (12, 30)
    assert x["imu_gl"].shape == (12, 6)
    assert x["imu_lh"].shape == (12, 6)
    assert x["airpods"].shape == (12, 6)

    np.testing.assert_allclose(x["imu_gl"], x["imu"][:, 0:6])
    np.testing.assert_allclose(x["imu_lh"], x["imu"][:, 6:12])
    np.testing.assert_allclose(x["imu_rh"], x["imu"][:, 12:18])
    np.testing.assert_allclose(x["imu_lp"], x["imu"][:, 18:24])
    np.testing.assert_allclose(x["imu_rp"], x["imu"][:, 24:30])

    raw_airpods = np.arange(12 * 9, dtype=np.float32).reshape(12, 9)
    np.testing.assert_allclose(x["airpods"], raw_airpods[:, 3:9])


def test_xrfv2_imu_normalization_variants() -> None:
    t30 = np.arange(12 * 30, dtype=np.float32).reshape(12, 30)
    from_t30 = XRFV2H5Adapter._normalize_imu_to_t30(t30)
    np.testing.assert_allclose(from_t30, t30)

    t56 = np.arange(12 * 5 * 6, dtype=np.float32).reshape(12, 5, 6)
    from_t56 = XRFV2H5Adapter._normalize_imu_to_t30(t56)
    np.testing.assert_allclose(from_t56, t56.reshape(12, 30))

    p56t = np.transpose(t56, (1, 2, 0))
    from_p56t = XRFV2H5Adapter._normalize_imu_to_t30(p56t)
    np.testing.assert_allclose(from_p56t, t56.reshape(12, 30))
