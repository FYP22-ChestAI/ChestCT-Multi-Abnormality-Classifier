"""The stage-3 interface: a bag of slice embeddings -> a volume embedding.

    forward(bags: (B, N, D) float, mask: (B, N) bool, True = real slice)  ->  AggregatorOutput
        pooled     (B, L)      one embedding per volume                     (per_label = False)
                   (B, C, L)   one embedding per volume AND label            (per_label = True)
        attention  (B, N) / (B, C, N)  weights over slices: >= 0, sum to 1 over real slices, 0 on padding
        instances  (B, N, L)   the per-slice features that were pooled

Bags come from ct_model.data.datasets.collate_bags, padded to the longest bag of the batch; padding
must never influence a result (a volume's output must not depend on which volumes share its batch).

``linear_pooling = True`` promises ``pooled == attention @ instances`` exactly. With a linear head
that makes every logit an exact sum of per-slice contributions (VolumeClassifier.contributions),
which is what the slice-level evidence uses. An aggregator that transforms after pooling sets it False.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AggregatorOutput:
    pooled: torch.Tensor
    attention: torch.Tensor | None = None
    instances: torch.Tensor | None = None


class Aggregator(nn.Module, ABC):
    per_label: bool = False
    linear_pooling: bool = False

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> AggregatorOutput:
        ...


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last axis (slices), with padded slices at exactly 0. ``mask`` broadcasts to
    ``scores`` ((B, N) against (B, ..., N)). Every bag has at least one real slice."""
    while mask.ndim < scores.ndim:
        mask = mask.unsqueeze(1)
    return torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1)
