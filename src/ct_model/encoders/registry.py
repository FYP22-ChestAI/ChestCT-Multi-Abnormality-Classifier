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
    """Build the SliceEncoder ``cfg`` describes, check its promised ``embed_dim``, wrap it with LoRA when
    ``cfg.lora`` is set (loading ``lora.adapter``), and (by default) freeze it -- only phase-2 training
    builds it with ``frozen=False``."""
    encoder = ENCODERS.get(cfg.type)(cfg)
    if cfg.embed_dim is not None and encoder.embed_dim != cfg.embed_dim:
        raise ValueError(f"encoder {cfg.name} produces {encoder.embed_dim}-d features but its config says embed_dim {cfg.embed_dim}")
    if cfg.lora is not None:
        from ct_model.training.records import file_sha256

        from .lora import LoRAConfig, apply_lora

        lora = LoRAConfig(**cfg.lora)
        encoder = apply_lora(encoder, lora)
        if lora.adapter and hasattr(encoder, "_provenance"):
            encoder._provenance.update(lora_adapter=str(lora.adapter), lora_adapter_sha256=file_sha256(lora.adapter))
    # frozen (encoding, inference): no gradients at all, also not into LoRA; frozen=False (phase-2 training):
    # with LoRA only its A / B matrices train (apply_lora froze the rest)
    return encoder.freeze() if frozen else encoder
