"""Fusion utilities for modality dropout and gating."""

from __future__ import annotations

from typing import Any

import numpy as np

ArrayDict = dict[str, np.ndarray]


def modality_dropout_mask(modalities: list[str], p: float, rng: np.random.Generator) -> dict[str, bool]:
    if p <= 0.0:
        return {m: True for m in modalities}
    if p >= 1.0:
        keep_one = modalities[int(rng.integers(0, len(modalities)))]
        return {m: (m == keep_one) for m in modalities}

    mask = {m: bool(rng.random() > p) for m in modalities}
    if not any(mask.values()):
        keep_one = modalities[int(rng.integers(0, len(modalities)))]
        mask[keep_one] = True
    return mask


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / np.sum(exp)


class GatingFusion:
    """Weighted-sum fusion across modality streams."""

    def __init__(self, modalities: list[str], seed: int = 42) -> None:
        self.modalities = list(modalities)
        rng = np.random.default_rng(seed)
        self.gate_logits = {m: float(rng.normal(0.0, 0.1)) for m in self.modalities}

    def fuse(
        self,
        features: ArrayDict,
        training: bool = False,
        dropout_p: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        if not features:
            raise ValueError("features cannot be empty")

        current_modalities = [m for m in self.modalities if m in features]
        if not current_modalities:
            raise ValueError("No expected modalities found in features dict")

        if training and dropout_p > 0.0:
            local_rng = rng if rng is not None else np.random.default_rng(0)
            mask = modality_dropout_mask(current_modalities, dropout_p, local_rng)
            active = [m for m in current_modalities if mask[m]]
        else:
            active = current_modalities

        if not active:
            raise ValueError("No modalities left after dropout")

        score_terms = []
        for modality in active:
            energy = float(np.mean(np.abs(features[modality])))
            score_terms.append(self.gate_logits[modality] + 0.05 * energy)
        scores = np.asarray(score_terms, dtype=np.float64)
        weights = _softmax(scores)

        fused = np.zeros_like(features[active[0]], dtype=np.float32)
        out_weights: dict[str, float] = {}
        for idx, modality in enumerate(active):
            weight = float(weights[idx])
            fused += weight * features[modality]
            out_weights[modality] = weight

        return fused, out_weights

    def state_dict(self) -> dict[str, Any]:
        return {"gate_logits": dict(self.gate_logits)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raw = state.get("gate_logits", {})
        for modality in self.modalities:
            if modality in raw:
                self.gate_logits[modality] = float(raw[modality])
