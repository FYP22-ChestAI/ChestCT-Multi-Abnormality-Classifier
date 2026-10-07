"""The baseline every aggregator must beat: the masked mean of the projected slice embeddings.

Same projection as ABMIL (Linear -> ReLU -> Dropout), so a comparison isolates the effect of
learned attention. Its "attention" is uniform, 1 / n_slices on every real slice.
"""
from __future__ import annotations

import torch
from torch import nn

from ..registry import AGGREGATORS
from .base import Aggregator, AggregatorOutput


class MeanPool(Aggregator):
    linear_pooling = True

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__(in_dim, hidden_dim)
        self.project = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> AggregatorOutput:
        h = self.project(bags)
        a = mask.to(h.dtype) / mask.sum(dim=1, keepdim=True).to(h.dtype)  # (B, N), 0 on padding
        return AggregatorOutput(pooled=(a.unsqueeze(1) @ h)[:, 0], attention=a, instances=h)


@AGGREGATORS.register("mean_pool")
def build_mean_pool(in_dim: int, n_labels: int, **params) -> MeanPool:
    return MeanPool(in_dim, **params)
