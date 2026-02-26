"""Tiny TCN-like baseline model."""

from __future__ import annotations

import numpy as np

from xrfv2_edge_tal.models.base_numpy import BaseNumpyFrameModel, moving_average


class TinyTCN(BaseNumpyFrameModel):
    model_name = "tiny_tcn"

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int = 32,
        kernel_size: int = 5,
        seed: int = 42,
    ) -> None:
        super().__init__(
            input_dims=input_dims, num_classes=num_classes, hidden_dim=hidden_dim, seed=seed
        )
        self.kernel_size = int(kernel_size)

    def _encode_modality(self, x: np.ndarray, modality: str) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        smoothed = moving_average(x, self.kernel_size)
        h = smoothed @ self.proj_w[modality] + self.proj_b[modality]
        return np.tanh(h)

    def _extra_state(self) -> dict[str, int]:
        return {"kernel_size": self.kernel_size}

    def _load_extra_state(self, state: dict[str, int]) -> None:
        if "kernel_size" in state:
            self.kernel_size = int(state["kernel_size"])
