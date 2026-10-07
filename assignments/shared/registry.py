"""Model registry, so a trainer builds models from configuration without importing them.

Each assignment keeps its own registry instance, because model names are only unique
within one assignment.

    models = Registry()

    @models.register("unet")
    def build_unet(**kwargs) -> nn.Module: ...

    model = models.build(config["model"]["name"], **config["model"]["args"])
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

ModelFactory = Callable[..., nn.Module]


class Registry:
    def __init__(self) -> None:
        self._factories: dict[str, ModelFactory] = {}

    def register(self, name: str) -> Callable[[ModelFactory], ModelFactory]:
        """Register a factory under `name`; duplicate names are a configuration error."""

        def decorator(factory: ModelFactory) -> ModelFactory:
            if name in self._factories:
                raise ValueError(f"model '{name}' is already registered")
            self._factories[name] = factory
            return factory

        return decorator

    def build(self, name: str, **kwargs) -> nn.Module:
        if name not in self._factories:
            raise KeyError(f"unknown model '{name}'; registered: {', '.join(self.names()) or 'none'}")
        return self._factories[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)
