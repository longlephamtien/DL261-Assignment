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

from shared.registry import ModelFactory, Registry  # noqa: F401  ModelFactory re-exported

_REGISTRY = Registry()


def register(name: str) -> Callable[[ModelFactory], ModelFactory]:
    """Register a factory under `name`; duplicate names are a configuration error."""
    return _REGISTRY.register(name)


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a registered model that maps images to raw logits of shape (N, 10)."""
    return _REGISTRY.build(name, **kwargs)


def list_models() -> list[str]:
    return _REGISTRY.names()


from . import linear, mlp, recurrent, transformer  # noqa: E402, F401  imported for their registration side effect
