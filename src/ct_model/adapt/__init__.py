"""Phase 2: LoRA adaptation of the slice encoder, end to end from the HU cache, with sharp / soft kernel
consistency (ct_model.adapt.trainer). Run with scripts/model/adapt_encoder.py; configs in configs/model/adapt/."""
from .config import AdaptConfig, load_adapt_config

__all__ = ["AdaptConfig", "load_adapt_config"]
