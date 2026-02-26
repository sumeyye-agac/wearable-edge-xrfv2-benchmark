"""Tiny Transformer-like baseline model."""

from __future__ import annotations

from typing import Any

import numpy as np

from xrfv2_edge_tal.models.base_numpy import BaseNumpyFrameModel


def _row_softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


class TinyTransformer(BaseNumpyFrameModel):
    model_name = "tiny_transformer"

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int = 32,
        seed: int = 42,
    ) -> None:
        super().__init__(
            input_dims=input_dims, num_classes=num_classes, hidden_dim=hidden_dim, seed=seed
        )
        self.wq = self.rng.normal(0.0, 0.1, size=(hidden_dim, hidden_dim)).astype(np.float32)
        self.wk = self.rng.normal(0.0, 0.1, size=(hidden_dim, hidden_dim)).astype(np.float32)
        self.wv = self.rng.normal(0.0, 0.1, size=(hidden_dim, hidden_dim)).astype(np.float32)

    def _encode_modality(self, x: np.ndarray, modality: str) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        h = np.tanh(x @ self.proj_w[modality] + self.proj_b[modality])
        q = h @ self.wq
        k = h @ self.wk
        v = h @ self.wv

        scale = float(np.sqrt(max(h.shape[1], 1)))
        attn = _row_softmax((q @ k.T) / scale)
        context = attn @ v
        return np.tanh(context)

    def _extra_state(self) -> dict[str, Any]:
        return {"wq": self.wq, "wk": self.wk, "wv": self.wv}

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        if "wq" in state:
            self.wq = np.asarray(state["wq"], dtype=np.float32)
        if "wk" in state:
            self.wk = np.asarray(state["wk"], dtype=np.float32)
        if "wv" in state:
            self.wv = np.asarray(state["wv"], dtype=np.float32)
