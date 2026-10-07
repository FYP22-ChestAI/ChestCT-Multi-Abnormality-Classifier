"""Encoder ``type`` (in configs/model/encoders/*.yaml) -> the function that builds it.

Adding a backbone that is a timm ViT needs only a new YAML file. A different family (a CNN, a 3D
model, a HF transformers model, ...) registers a builder:

    @register_encoder("my_type")
    def build_my_type(cfg: EncoderConfig) -> SliceEncoder: ...

and adds its module to ``ENCODERS`` in ct_model/registry.py.
"""
from __future__ import annotations

from ..config import EncoderConfig
from ..registry import ENCODERS


def register_encoder(type_name: str):
    return ENCODERS.register(type_name)


def available_encoder_types() -> list[str]:
    return ENCODERS.names()


def build_encoder(cfg: EncoderConfig, *, frozen: bool = True):
    """Build the SliceEncoder ``cfg`` describes, check its promised ``embed_dim``, and (by default)
    freeze it -- phase 1 never trains the encoder."""
    encoder = ENCODERS.get(cfg.type)(cfg)
    if cfg.embed_dim is not None and encoder.embed_dim != cfg.embed_dim:
        raise ValueError(f"encoder {cfg.name} produces {encoder.embed_dim}-d features but its config says embed_dim {cfg.embed_dim}")
    if cfg.lora is not None:
        from .lora import LoRAConfig, apply_lora

        encoder = apply_lora(encoder, LoRAConfig(**cfg.lora))
    return encoder.freeze() if frozen and cfg.lora is None else encoder
