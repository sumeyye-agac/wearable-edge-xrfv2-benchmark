from __future__ import annotations

import numpy as np

from xrfv2_edge_tal.event.preprocess import normalize_modalities


def test_normalize_modalities_per_sample() -> None:
    x = {
        "a": np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], dtype=np.float32),
        "b": np.array([[2.0], [2.0], [2.0]], dtype=np.float32),
    }
    out = normalize_modalities(x, enabled=True, clip=None)
    assert set(out.keys()) == {"a", "b"}
    assert out["a"].shape == (3, 2)
    assert abs(float(np.mean(out["a"][:, 0]))) < 1e-6
    assert abs(float(np.mean(out["a"][:, 1]))) < 1e-6
    # Constant feature stays finite after epsilon protection.
    assert np.all(np.isfinite(out["b"]))
