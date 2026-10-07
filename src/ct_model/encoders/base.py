"""The stage-2 interface every backbone implements.

A SliceEncoder turns raw HU slices into one feature vector per slice:

    forward(hu: (K, H, W) HU, any dtype)  ->  (K, embed_dim) float

It owns its input transform (InputTransform), so callers never window or normalise anything
themselves, and training, embedding extraction and inference cannot drift apart. Slice batching,
mixed precision and out-of-memory handling are the caller's job (ct_model.embeddings.extract), so
one encoder works for frozen extraction and for phase-2 LoRA training alike.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..config import EncoderConfig
from .transforms import InputTransform


class SliceEncoder(nn.Module, ABC):
    def __init__(self, cfg: EncoderConfig, input_transform: InputTransform, embed_dim: int):
        super().__init__()
        self.cfg = cfg
        self.input_transform = input_transform
        self.embed_dim = int(embed_dim)

    @property
    def name(self) -> str:
        return self.cfg.name

    def fingerprint(self) -> str:
        return self.cfg.fingerprint()

    @abstractmethod
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Model-ready input (K, C, h, w) -> (K, embed_dim)."""

    def forward(self, hu: torch.Tensor) -> torch.Tensor:
        return self.embed(self.input_transform(hu))

    def provenance(self) -> dict:
        """What else identifies these embeddings beyond the config (e.g. the resolved weights commit).
        Recorded in the embedding store."""
        return {}

    def freeze(self) -> "SliceEncoder":
        """Phase 1: no gradients anywhere, eval mode (no dropout / drop-path)."""
        for p in self.parameters():
            p.requires_grad_(False)
        return self.eval()
