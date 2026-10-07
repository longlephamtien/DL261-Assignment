"""Model registry.

Models are registered by name and built from configuration, so the trainer never
imports a model module directly.

    @register("unet")
    def build_unet(**kwargs) -> nn.Module: ...

    model = build_model("unet", **config["model"]["args"])
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from shared.registry import ModelFactory, Registry  # noqa: F401  ModelFactory re-exported

_REGISTRY = Registry()


def register(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a factory under `name`; duplicate names are a configuration error."""
    return _REGISTRY.register(name)


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a registered model that maps images to per-pixel logits of shape (N, 19, H, W)."""
    return _REGISTRY.build(name, **kwargs)


def list_models() -> list[str]:
    return _REGISTRY.names()
