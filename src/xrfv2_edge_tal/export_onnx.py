"""ONNX export utilities.

This module exports a torch-compatible graph that mirrors checkpointed numpy model
behavior for inference. It requires optional dependencies: torch and onnxruntime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from xrfv2_edge_tal.checkpoint import load_checkpoint
from xrfv2_edge_tal.models.factory import build_model


def _require_optional_deps() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError("ONNX export requires torch to be installed") from exc

    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError("ONNX export verification requires onnxruntime") from exc

    return torch, nn, ort


def _softmax_np(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _build_numpy_model(state: dict[str, Any], metadata: dict[str, Any], seed: int):
    model = build_model(
        name=str(metadata["model_name"]),
        input_dims=dict(metadata["input_dims"]),
        num_classes=int(metadata["num_classes"]),
        hidden_dim=int(metadata["hidden_dim"]),
        seed=seed,
        kernel_size=int(state.get("kernel_size", 5)),
        tcn_layers=int(state.get("tcn_layers", metadata.get("tcn_layers", 1))),
    )
    model.load_state_dict(state)
    return model


def export_onnx_main(
    checkpoint: str,
    config: dict[str, Any],
    output_path: str,
    seed: int = 42,
) -> Path:
    torch, nn, ort = _require_optional_deps()
    del config  # reserved for future export options

    state, metadata = load_checkpoint(checkpoint)
    numpy_model = _build_numpy_model(state=state, metadata=metadata, seed=seed)
    input_dims = dict(metadata["input_dims"])
    modalities = sorted(input_dims.keys())
    model_name = str(metadata["model_name"])

    class TorchWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.modalities = modalities
            self.model_name = model_name
            self.hidden_dim = int(metadata["hidden_dim"])
            self.num_classes = int(metadata["num_classes"])
            self.kernel_size = int(state.get("kernel_size", 5))
            self.tcn_layers = int(state.get("tcn_layers", 1))

            self.cls_w = nn.Parameter(
                torch.tensor(state["cls_w"], dtype=torch.float32), requires_grad=False
            )
            self.cls_b = nn.Parameter(
                torch.tensor(state["cls_b"], dtype=torch.float32), requires_grad=False
            )

            self.proj_w = nn.ParameterDict()
            self.proj_b = nn.ParameterDict()
            for modality in self.modalities:
                self.proj_w[modality] = nn.Parameter(
                    torch.tensor(state[f"proj_w::{modality}"], dtype=torch.float32),
                    requires_grad=False,
                )
                self.proj_b[modality] = nn.Parameter(
                    torch.tensor(state[f"proj_b::{modality}"], dtype=torch.float32),
                    requires_grad=False,
                )

            fusion = state.get("fusion", {}) if isinstance(state.get("fusion", {}), dict) else {}
            gate_logits = fusion.get("gate_logits", {}) if isinstance(fusion, dict) else {}
            self.gate_logits = {
                modality: float(gate_logits.get(modality, 0.0)) for modality in self.modalities
            }

            self.dw_kernels = nn.ParameterDict()
            if self.model_name == "tiny_tcn" and self.tcn_layers > 1:
                for modality in self.modalities:
                    for idx in range(self.tcn_layers - 1):
                        key = f"dw_kernel::{modality}::{idx}"
                        if key in state:
                            self.dw_kernels[f"{modality}__{idx}"] = nn.Parameter(
                                torch.tensor(state[key], dtype=torch.float32),
                                requires_grad=False,
                            )

            if self.model_name == "tiny_transformer":
                self.wq = nn.Parameter(
                    torch.tensor(state["wq"], dtype=torch.float32), requires_grad=False
                )
                self.wk = nn.Parameter(
                    torch.tensor(state["wk"], dtype=torch.float32), requires_grad=False
                )
                self.wv = nn.Parameter(
                    torch.tensor(state["wv"], dtype=torch.float32), requires_grad=False
                )

        def _moving_average(self, x: Any) -> Any:
            if self.kernel_size <= 1:
                return x
            x_t = x.transpose(1, 2)
            pad = self.kernel_size // 2
            x_pad = torch.nn.functional.pad(x_t, (pad, pad), mode="replicate")
            out = torch.nn.functional.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
            return out.transpose(1, 2)

        def _encode_modality(self, x: Any, modality: str) -> Any:
            if self.model_name == "tiny_tcn":
                h = self._moving_average(x)
                h = torch.tanh(h @ self.proj_w[modality] + self.proj_b[modality])
                if self.tcn_layers > 1:
                    pad = self.kernel_size // 2
                    for idx in range(self.tcn_layers - 1):
                        key = f"{modality}__{idx}"
                        if key not in self.dw_kernels:
                            continue
                        k = self.dw_kernels[key]
                        ht = h.transpose(1, 2)
                        hpad = torch.nn.functional.pad(ht, (pad, pad), mode="replicate")
                        conv = torch.nn.functional.conv1d(
                            hpad,
                            k,
                            stride=1,
                            padding=0,
                            groups=self.hidden_dim,
                        )
                        h = torch.tanh(conv.transpose(1, 2) + h)
                return h

            h = torch.tanh(x @ self.proj_w[modality] + self.proj_b[modality])
            q = h @ self.wq
            k = h @ self.wk
            v = h @ self.wv
            scale = float(np.sqrt(max(self.hidden_dim, 1)))
            attn = torch.softmax((q @ k.transpose(1, 2)) / scale, dim=-1)
            return torch.tanh(attn @ v)

        def forward(self, *inputs: Any) -> Any:
            feats = []
            score_terms = []
            for idx, modality in enumerate(self.modalities):
                feat = self._encode_modality(inputs[idx], modality)
                feats.append(feat)
                energy = torch.mean(torch.abs(feat))
                score_terms.append(
                    torch.tensor(self.gate_logits[modality], dtype=feat.dtype, device=feat.device)
                    + 0.05 * energy
                )

            score_vec = torch.stack(score_terms)
            weights = torch.softmax(score_vec, dim=0)

            fused = torch.zeros_like(feats[0])
            for idx, feat in enumerate(feats):
                fused = fused + weights[idx] * feat

            logits = fused @ self.cls_w + self.cls_b
            probs = torch.softmax(logits, dim=-1)
            return probs

    wrapper = TorchWrapper().eval()

    seq_len = 160
    rng = np.random.default_rng(seed)
    np_inputs = {
        modality: rng.normal(0.0, 1.0, size=(1, seq_len, input_dims[modality])).astype(np.float32)
        for modality in modalities
    }

    with torch.no_grad():
        torch_inputs = [torch.tensor(np_inputs[m], dtype=torch.float32) for m in modalities]
        torch_out = wrapper(*torch_inputs).cpu().numpy()

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_names = [f"input_{m}" for m in modalities]
    dynamic_axes = {name: {1: "time"} for name in input_names}
    dynamic_axes["probs"] = {1: "time"}

    torch.onnx.export(
        wrapper,
        args=tuple(torch_inputs),
        f=str(out_path),
        input_names=input_names,
        output_names=["probs"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )

    ort_session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: np_inputs[modality] for name, modality in zip(input_names, modalities, strict=False)
    }
    ort_out = ort_session.run(["probs"], ort_inputs)[0]

    np_out = numpy_model.predict_proba({m: np_inputs[m][0] for m in modalities})
    np_out = np_out[None, ...]

    if not np.allclose(ort_out, torch_out, atol=1e-4, rtol=1e-4):
        raise RuntimeError(
            "ONNX verification failed: ONNXRuntime output diverges from torch output"
        )
    if not np.allclose(ort_out, np_out, atol=2e-3, rtol=2e-3):
        raise RuntimeError(
            "ONNX verification failed: ONNXRuntime output diverges from numpy checkpoint model"
        )

    return out_path
