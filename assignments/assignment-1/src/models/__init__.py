"""Model registry.

Models are registered by name and built from configuration, so the trainer never
imports a model module directly.

    @register("linear")
    def build_linear(**kwargs) -> nn.Module: ...

    model = build_model("linear", **config["model"]["args"])
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

ModelFactory = Callable[..., nn.Module]

_REGISTRY: dict[str, ModelFactory] = {}


def register(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a factory under `name`; duplicate names are a configuration error."""

    def decorator(factory: ModelFactory) -> ModelFactory:
        if name in _REGISTRY:
            raise ValueError(f"model '{name}' is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a registered model that maps images to raw logits of shape (N, 10)."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'; registered: {', '.join(list_models()) or 'none'}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    return sorted(_REGISTRY)


from . import linear, mlp

