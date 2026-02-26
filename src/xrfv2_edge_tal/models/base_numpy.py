"""Numpy model base class for frame-wise classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np

from xrfv2_edge_tal.models.fusion import GatingFusion

ArrayDict = dict[str, np.ndarray]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


def moving_average(x: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(x)
    for t in range(x.shape[0]):
        out[t] = np.mean(padded[t : t + kernel_size], axis=0)
    return out


class BaseNumpyFrameModel:
    model_name = "base_numpy"

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.input_dims = dict(input_dims)
        self.modalities = sorted(list(input_dims.keys()))
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.rng = np.random.default_rng(seed)

        self.proj_w: dict[str, np.ndarray] = {}
        self.proj_b: dict[str, np.ndarray] = {}
        for modality, dim in self.input_dims.items():
            self.proj_w[modality] = self.rng.normal(0.0, 0.1, size=(dim, hidden_dim)).astype(np.float32)
            self.proj_b[modality] = np.zeros((hidden_dim,), dtype=np.float32)

        self.cls_w = self.rng.normal(0.0, 0.1, size=(hidden_dim, num_classes)).astype(np.float32)
        self.cls_b = np.zeros((num_classes,), dtype=np.float32)

        self.fusion = GatingFusion(self.modalities, seed=seed)

    def _encode_modality(self, x: np.ndarray, modality: str) -> np.ndarray:
        raise NotImplementedError

    def _forward_features(
        self,
        x_dict: ArrayDict,
        training: bool = False,
        modality_dropout_p: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, float]]:
        feats: dict[str, np.ndarray] = {}
        for modality in self.modalities:
            if modality not in x_dict:
                continue
            feats[modality] = self._encode_modality(x_dict[modality], modality)

        fused, weights = self.fusion.fuse(feats, training=training, dropout_p=modality_dropout_p, rng=self.rng)
        return fused, weights

    def forward(
        self,
        x_dict: ArrayDict,
        training: bool = False,
        modality_dropout_p: float = 0.0,
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
        fused, weights = self._forward_features(
            x_dict=x_dict,
            training=training,
            modality_dropout_p=modality_dropout_p,
        )
        logits = fused @ self.cls_w + self.cls_b
        return logits, weights, fused

    def predict_proba(self, x_dict: ArrayDict) -> np.ndarray:
        logits, _, _ = self.forward(x_dict, training=False, modality_dropout_p=0.0)
        return softmax(logits)

    def train_step(
        self,
        x_dict: ArrayDict,
        target: np.ndarray,
        lr: float,
        modality_dropout_p: float = 0.0,
        teacher_probs: np.ndarray | None = None,
        distill_weight: float = 0.0,
        temperature: float = 2.0,
        focal_gamma: float = 0.0,
        background_label: int = 0,
        background_weight: float = 1.0,
        class_balance: bool = False,
        window_pooling: str | None = None,
    ) -> float:
        logits, _, fused = self.forward(x_dict, training=True, modality_dropout_p=modality_dropout_p)
        if window_pooling is not None and int(logits.shape[0]) > 1:
            mode = str(window_pooling).strip().lower()
            if mode == "max":
                logits = np.max(logits, axis=0, keepdims=True)
                fused = np.mean(fused, axis=0, keepdims=True)
            elif mode == "mean":
                logits = np.mean(logits, axis=0, keepdims=True)
                fused = np.mean(fused, axis=0, keepdims=True)
            else:
                raise ValueError(
                    f"Unsupported window_pooling={window_pooling}. Expected: max|mean."
                )
        probs = softmax(logits)

        t = target.astype(np.int64)
        t = np.clip(t, 0, self.num_classes - 1)
        if len(t) != logits.shape[0]:
            if len(t) == 0:
                raise ValueError("target cannot be empty")
            t = t[:1]

        n = max(len(t), 1)
        ce = -np.log(np.maximum(probs[np.arange(len(t)), t], 1e-12))

        weights = np.ones((len(t),), dtype=np.float32)
        if 0 <= int(background_label) < self.num_classes and background_weight != 1.0:
            weights[t == int(background_label)] *= float(background_weight)
        if class_balance and len(t) > 0:
            classes, counts = np.unique(t, return_counts=True)
            inv = {int(c): 1.0 / float(max(cnt, 1)) for c, cnt in zip(classes, counts, strict=True)}
            mean_inv = float(np.mean(list(inv.values()))) if inv else 1.0
            for cls, inv_w in inv.items():
                weights[t == cls] *= float(inv_w / max(mean_inv, 1e-12))

        if focal_gamma > 0.0:
            pt = np.maximum(probs[np.arange(len(t)), t], 1e-12)
            ce = ((1.0 - pt) ** float(focal_gamma)) * ce

        loss = float(np.sum(ce * weights) / max(np.sum(weights), 1e-9))

        grad_ce = probs.copy()
        grad_ce[np.arange(len(t)), t] -= 1.0
        grad_ce *= weights[:, None]
        grad_ce /= max(np.sum(weights), 1e-9)

        grad = grad_ce
        kd_loss = 0.0
        w = float(np.clip(distill_weight, 0.0, 1.0))
        if teacher_probs is not None and teacher_probs.shape == probs.shape and w > 0.0:
            t_probs = np.asarray(teacher_probs, dtype=np.float32)
            t_probs = np.maximum(t_probs, 1e-12)
            t_probs = t_probs / np.sum(t_probs, axis=1, keepdims=True)

            student_temp = softmax(logits / max(temperature, 1e-6))
            log_student = np.log(np.maximum(student_temp, 1e-12))
            log_teacher = np.log(np.maximum(t_probs, 1e-12))
            kd_loss = float(np.mean(np.sum(t_probs * (log_teacher - log_student), axis=1)) * (temperature**2))

            grad_kd = (student_temp - t_probs) / n
            grad = (1.0 - w) * grad_ce + w * grad_kd

        grad_w = fused.T @ grad
        grad_b = np.sum(grad, axis=0)

        self.cls_w -= lr * grad_w.astype(np.float32)
        self.cls_b -= lr * grad_b.astype(np.float32)

        return float((1.0 - w) * loss + w * kd_loss)

    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "modalities": list(self.modalities),
            "cls_w": self.cls_w,
            "cls_b": self.cls_b,
            "fusion": self.fusion.state_dict(),
        }
        for modality in self.modalities:
            out[f"proj_w::{modality}"] = self.proj_w[modality]
            out[f"proj_b::{modality}"] = self.proj_b[modality]
        out.update(self._extra_state())
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.cls_w = np.asarray(state["cls_w"], dtype=np.float32)
        self.cls_b = np.asarray(state["cls_b"], dtype=np.float32)
        for modality in self.modalities:
            self.proj_w[modality] = np.asarray(state[f"proj_w::{modality}"], dtype=np.float32)
            self.proj_b[modality] = np.asarray(state[f"proj_b::{modality}"], dtype=np.float32)
        fusion_state = state.get("fusion", {})
        if isinstance(fusion_state, dict):
            self.fusion.load_state_dict(fusion_state)
        self._load_extra_state(state)

    def _extra_state(self) -> dict[str, Any]:
        return {}

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        _ = state


__all__ = ["BaseNumpyFrameModel", "softmax", "moving_average"]
