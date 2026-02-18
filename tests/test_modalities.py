from __future__ import annotations

import numpy as np
import pytest

from xrfv2_edge_tal.data.adapters import DummyAdapter
from xrfv2_edge_tal.modalities import (
    mask_channels_by_profile,
    normalize_modality_name,
    resolve_requested_modalities,
    stack_modalities_with_channel_names,
)


def test_normalize_modality_aliases() -> None:
    assert normalize_modality_name("airpods") == "earbuds"
    assert normalize_modality_name("imu_glasses") == "glasses"
    assert normalize_modality_name("smartwatch") == "watch"
    assert normalize_modality_name("mobile") == "phone"


def test_resolve_requested_modalities_missing_is_explicit() -> None:
    available = ["imu_phone", "imu_watch", "imu_earbuds"]
    with pytest.raises(ValueError, match="Available modalities"):
        resolve_requested_modalities(available, ["glasses"])


def test_stack_and_mask_channels_by_profile() -> None:
    x = {
        "imu_earbuds": np.ones((6, 3), dtype=np.float32),
        "imu_glasses": 2.0 * np.ones((6, 3), dtype=np.float32),
        "imu_watch": 3.0 * np.ones((6, 3), dtype=np.float32),
    }
    stacked, names = stack_modalities_with_channel_names(x)
    assert stacked.shape == (6, 9)
    assert names[0].startswith("earbuds:")

    masked, masked_names = mask_channels_by_profile(
        x=stacked,
        channel_names=names,
        include_modalities=["earbuds", "glasses"],
    )
    assert masked.shape == (6, 6)
    assert all(name.split(":", 1)[0] in {"earbuds", "glasses"} for name in masked_names)


def test_dummy_profile_masking_end_to_end() -> None:
    adapter = DummyAdapter(seed=3, num_train=2, num_test=1)
    x, _, _ = adapter.get_sample("0", "train")
    stacked, names = stack_modalities_with_channel_names(x)

    masked, masked_names = mask_channels_by_profile(
        x=stacked,
        channel_names=names,
        include_modalities=["glasses"],
    )
    assert masked.shape[0] == stacked.shape[0]
    assert 0 < masked.shape[1] < stacked.shape[1]
    assert all(name.startswith("glasses:") for name in masked_names)
