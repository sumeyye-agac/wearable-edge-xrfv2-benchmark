"""Torch-backed baseline models with optional MPS acceleration."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ArrayDict = dict[str, np.ndarray]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class TorchBaseFrameModel:
    model_name = "torch_base"

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int = 32,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.input_dims = dict(input_dims)
        self.modalities = sorted(list(input_dims.keys()))
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.device = _resolve_device(device)

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.proj_w: dict[str, torch.Tensor] = {}
        self.proj_b: dict[str, torch.Tensor] = {}
        for modality, dim in self.input_dims.items():
            self.proj_w[modality] = (
                0.1 * torch.randn(dim, hidden_dim, device=self.device, dtype=torch.float32)
            ).requires_grad_(True)
            self.proj_b[modality] = torch.zeros(
                hidden_dim, device=self.device, dtype=torch.float32, requires_grad=True
            )

        self.cls_w = (
            0.1 * torch.randn(hidden_dim, num_classes, device=self.device, dtype=torch.float32)
        ).requires_grad_(True)
        self.cls_b = torch.zeros(
            num_classes, device=self.device, dtype=torch.float32, requires_grad=True
        )
        self.gate_logits: dict[str, torch.Tensor] = {
            m: (0.1 * torch.randn((), device=self.device, dtype=torch.float32)).requires_grad_(True)
            for m in self.modalities
        }

    def _parameters(self) -> list[torch.Tensor]:
        out = [self.cls_w, self.cls_b]
        for modality in self.modalities:
            out.append(self.proj_w[modality])
            out.append(self.proj_b[modality])
            out.append(self.gate_logits[modality])
        out.extend(self._extra_parameters())
        return out

    def _extra_parameters(self) -> list[torch.Tensor]:
        return []

    def _encode_modality(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        raise NotImplementedError

    def _dropout_mask(self, p: float) -> dict[str, bool]:
        if p <= 0.0:
            return {m: True for m in self.modalities}
        if p >= 1.0:
            keep = self.modalities[int(self.rng.integers(0, len(self.modalities)))]
            return {m: (m == keep) for m in self.modalities}

        mask = {m: bool(self.rng.random() > p) for m in self.modalities}
        if not any(mask.values()):
            keep = self.modalities[int(self.rng.integers(0, len(self.modalities)))]
            mask[keep] = True
        return mask

    def _to_torch_inputs(self, x_dict: ArrayDict) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for modality in self.modalities:
            if modality not in x_dict:
                continue
            out[modality] = torch.as_tensor(
                x_dict[modality], device=self.device, dtype=torch.float32
            )
        return out

    def forward(
        self,
        x_dict: ArrayDict,
        training: bool = False,
        modality_dropout_p: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        x_torch = self._to_torch_inputs(x_dict)
        feats: dict[str, torch.Tensor] = {}
        for modality, x in x_torch.items():
            feats[modality] = self._encode_modality(x, modality)

        available = [m for m in self.modalities if m in feats]
        if not available:
            raise ValueError("No modalities available for forward pass")

        if training and modality_dropout_p > 0.0:
            mask = self._dropout_mask(modality_dropout_p)
            active = [m for m in available if mask[m]]
            if not active:
                active = [available[0]]
        else:
            active = available

        scores = torch.stack(
            [self.gate_logits[m] + 0.05 * torch.mean(torch.abs(feats[m])) for m in active]
        )
        weights = torch.softmax(scores, dim=0)

        fused = torch.zeros_like(feats[active[0]])
        weight_map: dict[str, float] = {}
        for idx, modality in enumerate(active):
            fused = fused + weights[idx] * feats[modality]
            weight_map[modality] = float(weights[idx].detach().cpu().item())

        logits = fused @ self.cls_w + self.cls_b
        return logits, weight_map, fused

    def predict_proba(self, x_dict: ArrayDict) -> np.ndarray:
        with torch.no_grad():
            logits, _, _ = self.forward(x_dict, training=False, modality_dropout_p=0.0)
            probs = torch.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

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
        logits, _, _ = self.forward(x_dict, training=True, modality_dropout_p=modality_dropout_p)
        if window_pooling is not None and int(logits.shape[0]) > 1:
            mode = str(window_pooling).strip().lower()
            if mode == "max":
                logits = torch.max(logits, dim=0, keepdim=True).values
            elif mode == "mean":
                logits = torch.mean(logits, dim=0, keepdim=True)
            else:
                raise ValueError(
                    f"Unsupported window_pooling={window_pooling}. Expected: max|mean."
                )

        target_t = torch.as_tensor(target, device=self.device, dtype=torch.long)
        target_t = torch.clamp(target_t, min=0, max=self.num_classes - 1)
        if int(target_t.numel()) != int(logits.shape[0]):
            if int(target_t.numel()) == 0:
                raise ValueError("target cannot be empty")
            target_t = target_t[:1]
        ce_per = F.cross_entropy(logits, target_t, reduction="none")

        weights = torch.ones_like(ce_per)
        bg = int(background_label)
        if 0 <= bg < self.num_classes and background_weight != 1.0:
            weights = torch.where(
                target_t == bg,
                weights * float(background_weight),
                weights,
            )
        if class_balance and int(target_t.numel()) > 0:
            classes, counts = torch.unique(target_t, return_counts=True)
            inv = 1.0 / torch.clamp(counts.float(), min=1.0)
            inv = inv / torch.clamp(torch.mean(inv), min=1e-12)
            class_weights = torch.ones((self.num_classes,), device=self.device, dtype=torch.float32)
            class_weights[classes] = inv
            weights = weights * class_weights[target_t]

        if focal_gamma > 0.0:
            probs = torch.softmax(logits, dim=1)
            pt = probs[torch.arange(probs.shape[0], device=self.device), target_t]
            ce_per = ((1.0 - pt) ** float(focal_gamma)) * ce_per

        ce_loss = torch.sum(ce_per * weights) / torch.clamp(torch.sum(weights), min=1e-9)

        loss = ce_loss
        w = float(np.clip(distill_weight, 0.0, 1.0))
        if teacher_probs is not None and teacher_probs.shape == tuple(logits.shape) and w > 0.0:
            t_probs = torch.as_tensor(teacher_probs, device=self.device, dtype=torch.float32)
            t_probs = torch.clamp(t_probs, min=1e-12)
            t_probs = t_probs / torch.sum(t_probs, dim=1, keepdim=True)

            student_temp = torch.softmax(logits / max(temperature, 1e-6), dim=1)
            log_student = torch.log(torch.clamp(student_temp, min=1e-12))
            log_teacher = torch.log(torch.clamp(t_probs, min=1e-12))
            kd_loss = torch.mean(
                torch.sum(t_probs * (log_teacher - log_student), dim=1) * (temperature**2)
            )
            loss = (1.0 - w) * ce_loss + w * kd_loss

        for p in self._parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()

        with torch.no_grad():
            for p in self._parameters():
                if p.grad is not None:
                    p -= lr * p.grad

        return float(loss.detach().cpu().item())

    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "modalities": list(self.modalities),
            "cls_w": self.cls_w.detach().cpu().numpy(),
            "cls_b": self.cls_b.detach().cpu().numpy(),
            "fusion": {
                "gate_logits": {
                    m: float(self.gate_logits[m].detach().cpu().item()) for m in self.modalities
                }
            },
        }
        for modality in self.modalities:
            out[f"proj_w::{modality}"] = self.proj_w[modality].detach().cpu().numpy()
            out[f"proj_b::{modality}"] = self.proj_b[modality].detach().cpu().numpy()
        out.update(self._extra_state())
        return out

    def load_state_dict(self, state: dict[str, Any]) -> None:
        with torch.no_grad():
            self.cls_w.copy_(
                torch.as_tensor(state["cls_w"], device=self.device, dtype=torch.float32)
            )
            self.cls_b.copy_(
                torch.as_tensor(state["cls_b"], device=self.device, dtype=torch.float32)
            )

            for modality in self.modalities:
                self.proj_w[modality].copy_(
                    torch.as_tensor(
                        state[f"proj_w::{modality}"], device=self.device, dtype=torch.float32
                    )
                )
                self.proj_b[modality].copy_(
                    torch.as_tensor(
                        state[f"proj_b::{modality}"], device=self.device, dtype=torch.float32
                    )
                )

            fusion = state.get("fusion", {})
            gate_logits = fusion.get("gate_logits", {}) if isinstance(fusion, dict) else {}
            for modality in self.modalities:
                if modality in gate_logits:
                    self.gate_logits[modality].copy_(
                        torch.as_tensor(
                            gate_logits[modality], device=self.device, dtype=torch.float32
                        )
                    )

        self._load_extra_state(state)

    def _extra_state(self) -> dict[str, Any]:
        return {}

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        _ = state


class TorchTinyTCN(TorchBaseFrameModel):
    model_name = "tiny_tcn"

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int = 32,
        kernel_size: int = 5,
        tcn_layers: int = 1,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            seed=seed,
            device=device,
        )
        self.kernel_size = int(kernel_size)
        self.tcn_layers = max(1, int(tcn_layers))
        self.dw_kernels: dict[str, list[torch.Tensor]] = {}
        for modality in self.modalities:
            self.dw_kernels[modality] = []
            for _ in range(self.tcn_layers - 1):
                k = torch.zeros(
                    (self.hidden_dim, 1, self.kernel_size),
                    device=self.device,
                    dtype=torch.float32,
                )
                center = self.kernel_size // 2
                k[:, 0, center] = 1.0
                k = (k + 0.01 * torch.randn_like(k)).requires_grad_(True)
                self.dw_kernels[modality].append(k)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size <= 1:
            return x
        xt = x.transpose(0, 1).unsqueeze(0)
        pad = self.kernel_size // 2
        xpad = F.pad(xt, (pad, pad), mode="replicate")
        out = F.avg_pool1d(xpad, kernel_size=self.kernel_size, stride=1)
        return out.squeeze(0).transpose(0, 1)

    def _encode_modality(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        smoothed = self._moving_average(x)
        h = torch.tanh(smoothed @ self.proj_w[modality] + self.proj_b[modality])
        if self.tcn_layers <= 1:
            return h
        pad = self.kernel_size // 2
        for kernel in self.dw_kernels.get(modality, []):
            ht = h.transpose(0, 1).unsqueeze(0)
            hpad = F.pad(ht, (pad, pad), mode="replicate")
            conv = F.conv1d(hpad, kernel, stride=1, padding=0, groups=self.hidden_dim)
            conv_t = conv.squeeze(0).transpose(0, 1)
            h = torch.tanh(conv_t + h)
        return h

    def _extra_parameters(self) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for modality in self.modalities:
            out.extend(self.dw_kernels.get(modality, []))
        return out

    def _extra_state(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kernel_size": self.kernel_size, "tcn_layers": self.tcn_layers}
        for modality in self.modalities:
            for idx, kernel in enumerate(self.dw_kernels.get(modality, [])):
                out[f"dw_kernel::{modality}::{idx}"] = kernel.detach().cpu().numpy()
        return out

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        if "kernel_size" in state:
            self.kernel_size = int(state["kernel_size"])
        if "tcn_layers" in state:
            self.tcn_layers = int(state["tcn_layers"])
        for modality in self.modalities:
            for idx, kernel in enumerate(self.dw_kernels.get(modality, [])):
                key = f"dw_kernel::{modality}::{idx}"
                if key in state:
                    with torch.no_grad():
                        kernel.copy_(
                            torch.as_tensor(state[key], device=self.device, dtype=torch.float32)
                        )


class TorchTinyTransformer(TorchBaseFrameModel):
    model_name = "tiny_transformer"

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int = 32,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        super().__init__(
            input_dims=input_dims,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            seed=seed,
            device=device,
        )
        self.wq = (
            0.1 * torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=torch.float32)
        ).requires_grad_(True)
        self.wk = (
            0.1 * torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=torch.float32)
        ).requires_grad_(True)
        self.wv = (
            0.1 * torch.randn(hidden_dim, hidden_dim, device=self.device, dtype=torch.float32)
        ).requires_grad_(True)

    def _extra_parameters(self) -> list[torch.Tensor]:
        return [self.wq, self.wk, self.wv]

    def _encode_modality(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        h = torch.tanh(x @ self.proj_w[modality] + self.proj_b[modality])
        q = h @ self.wq
        k = h @ self.wk
        v = h @ self.wv
        scale = float(np.sqrt(max(self.hidden_dim, 1)))
        attn = torch.softmax((q @ k.transpose(0, 1)) / scale, dim=1)
        return torch.tanh(attn @ v)

    def _extra_state(self) -> dict[str, np.ndarray]:
        return {
            "wq": self.wq.detach().cpu().numpy(),
            "wk": self.wk.detach().cpu().numpy(),
            "wv": self.wv.detach().cpu().numpy(),
        }

    def _load_extra_state(self, state: dict[str, Any]) -> None:
        with torch.no_grad():
            if "wq" in state:
                self.wq.copy_(torch.as_tensor(state["wq"], device=self.device, dtype=torch.float32))
            if "wk" in state:
                self.wk.copy_(torch.as_tensor(state["wk"], device=self.device, dtype=torch.float32))
            if "wv" in state:
                self.wv.copy_(torch.as_tensor(state["wv"], device=self.device, dtype=torch.float32))
