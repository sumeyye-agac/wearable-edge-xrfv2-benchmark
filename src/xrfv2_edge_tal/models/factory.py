"""Model factory."""

from __future__ import annotations

from typing import Any

from xrfv2_edge_tal.models.tiny_tcn import TinyTCN
from xrfv2_edge_tal.models.tiny_transformer import TinyTransformer


def build_model(
    name: str,
    input_dims: dict[str, int],
    num_classes: int,
    hidden_dim: int = 32,
    seed: int = 42,
    backend: str = "numpy",
    device: str = "auto",
    **kwargs: Any,
) -> Any:
    model_name = name.lower()
    backend_name = backend.lower()

    if backend_name == "torch":
        try:
            from xrfv2_edge_tal.models.torch_models import TorchTinyTCN, TorchTinyTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Torch backend requested but torch is not installed. Install torch first."
            ) from exc

        if model_name == "tiny_tcn":
            kernel_size = int(kwargs.get("kernel_size", 5))
            return TorchTinyTCN(
                input_dims=input_dims,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                seed=seed,
                device=device,
            )
        if model_name == "tiny_transformer":
            return TorchTinyTransformer(
                input_dims=input_dims,
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                seed=seed,
                device=device,
            )
        raise ValueError(f"Unknown model name: {name}")

    if model_name == "tiny_tcn":
        kernel_size = int(kwargs.get("kernel_size", 5))
        return TinyTCN(
            input_dims=input_dims,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            seed=seed,
        )
    if model_name == "tiny_transformer":
        return TinyTransformer(
            input_dims=input_dims,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            seed=seed,
        )
    raise ValueError(f"Unknown model name: {name}")
