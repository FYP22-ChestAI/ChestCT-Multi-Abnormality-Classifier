"""Name -> builder registries, so every swappable component is chosen by a ``type:`` in a YAML file.

    ENCODERS     ct_model.encoders      (``type`` in configs/model/encoders/*.yaml)
    AGGREGATORS  ct_model.aggregators   (``aggregator.type`` in configs/model/experiments/*.yaml)
    HEADS        ct_model.heads         (``head.type``)
    LOSSES       ct_model.training.losses (``loss.type``)

A new component is one decorated function in its package (``@AGGREGATORS.register("my_mil")``) plus
its module in the registry's ``builtin`` list, so it is found without being imported first.
"""
from __future__ import annotations

import importlib
from typing import Callable


class Registry:
    def __init__(self, kind: str, builtin: tuple[str, ...] = ()):
        self.kind = kind
        self._builtin = builtin
        self._loaded = False
        self._builders: dict[str, Callable] = {}

    def register(self, name: str) -> Callable[[Callable], Callable]:
        def decorator(fn: Callable) -> Callable:
            if name in self._builders and self._builders[name] is not fn:
                raise ValueError(f"{self.kind} type {name!r} is registered twice")
            self._builders[name] = fn
            return fn

        return decorator

    def _load_builtin(self) -> None:
        if not self._loaded:
            self._loaded = True
            for module in self._builtin:
                importlib.import_module(module)

    def names(self) -> list[str]:
        self._load_builtin()
        return sorted(self._builders)

    def get(self, name: str) -> Callable:
        self._load_builtin()
        if name not in self._builders:
            raise ValueError(f"unknown {self.kind} type {name!r}; registered: {sorted(self._builders)}")
        return self._builders[name]

    def build(self, spec: dict, **context):
        """Build from a config dict ``{"type": name, **params}``. ``context`` holds what the caller
        knows (input size, number of labels, ...). A parameter the builder does not take is an error,
        like an unknown key anywhere else in the configs."""
        spec = dict(spec or {})
        if "type" not in spec:
            raise ValueError(f"{self.kind} config needs a 'type' (one of {self.names()})")
        name = spec.pop("type")
        try:
            return self.get(name)(**context, **spec)
        except TypeError as exc:
            raise ValueError(f"invalid {self.kind} config for type {name!r}: {exc}") from exc


ENCODERS = Registry("encoder", ("ct_model.encoders.timm_vit",))
AGGREGATORS = Registry("aggregator", (
    "ct_model.aggregators.abmil", "ct_model.aggregators.mean_pool", "ct_model.aggregators.query_mil",
))
HEADS = Registry("head", ("ct_model.heads.linear",))
LOSSES = Registry("loss", ("ct_model.training.losses",))
