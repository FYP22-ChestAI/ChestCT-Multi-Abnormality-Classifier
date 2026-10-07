"""The stage-4 interface.

    forward(pooled: (B, in_dim), or (B, Q, in_dim) from a query-based aggregator)  ->  logits (B, n_labels)

Logits, never probabilities: the loss (BCE-with-logits / ASL) applies the sigmoid itself, which is
numerically safer. Label order is ct_model.data.volumes.label_names(manifest), saved with every checkpoint.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class Head(nn.Module, ABC):
    def __init__(self, in_dim: int, n_labels: int):
        super().__init__()
        self.in_dim = in_dim
        self.n_labels = n_labels

    @abstractmethod
    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        ...
