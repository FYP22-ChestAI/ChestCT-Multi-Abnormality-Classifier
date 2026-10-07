"""Encoder ``type`` (in configs/model/encoders/*.yaml) -> the function that builds it.

Adding a backbone that is a timm ViT needs only a new YAML file. A different family (a CNN, a 3D
model, a HF transformers model, ...) registers a builder here:

    @register_encoder("my_type")
    def build_my_type(cfg: EncoderConfig) -> SliceEncoder: ...

and imports its module in ``_load_builtin`` below.
"""
from __future__ import annotations

from typing import Callable

from ..config import EncoderConfig

_BUILDERS: dict[str, Callable] = {}


def register_encoder(type_name: str):
    def decorator(fn: Callable) -> Callable:
        if type_name in _BUILDERS and _BUILDERS[type_name] is not fn:
            raise ValueError(f"encoder type {type_name!r} is registered twice")
        _BUILDERS[type_name] = fn
        return fn

    return decorator


def _load_builtin() -> None:
    from . import timm_vit  # noqa: F401 - registers "timm_vit"


def available_encoder_types() -> list[str]:
    _load_builtin()
    return sorted(_BUILDERS)


def build_encoder(cfg: EncoderConfig, *, frozen: bool = True):
    """Build the SliceEncoder ``cfg`` describes, check its promised ``embed_dim``, and (by default)
    freeze it -- phase 1 never trains the encoder."""
    _load_builtin()
    if cfg.type not in _BUILDERS:
        raise ValueError(f"unknown encoder type {cfg.type!r}; registered: {sorted(_BUILDERS)}")
    encoder = _BUILDERS[cfg.type](cfg)
    if cfg.embed_dim is not None and encoder.embed_dim != cfg.embed_dim:
        raise ValueError(f"encoder {cfg.name} produces {encoder.embed_dim}-d features but its config says embed_dim {cfg.embed_dim}")
    if cfg.lora is not None:
        from .lora import LoRAConfig, apply_lora

        encoder = apply_lora(encoder, LoRAConfig(**cfg.lora))
    return encoder.freeze() if frozen and cfg.lora is None else encoder
