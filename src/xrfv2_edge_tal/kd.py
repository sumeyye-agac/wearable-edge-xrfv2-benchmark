"""Knowledge distillation helpers."""

from __future__ import annotations

import numpy as np


def safe_normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.maximum(probs, 1e-12)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs.astype(np.float32)
