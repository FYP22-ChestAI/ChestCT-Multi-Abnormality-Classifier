"""Stage 4: volume embedding -> multi-label logits (one per abnormality). Built by ``head.type``
(ct_model.registry.HEADS): ``linear`` (linear.py)."""
from .base import Head

__all__ = ["Head"]
