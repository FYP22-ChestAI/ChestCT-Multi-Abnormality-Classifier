"""The stage-3 interface.

    forward(bags: (B, N, D) float, mask: (B, N) bool, True = real slice)  ->  AggregatorOutput
        pooled     (B, out_dim), or (B, Q, out_dim) for query-based MIL -- one embedding per volume (per query)
        attention  (B, N) or (B, Q, N) -- weights over slices, 0 on padding; None if the method has none

Bags come from ct_model.data.datasets.collate_bags (padded to the longest bag in the batch). Row n of a
bag is cache slice ``slice_idx[b, n]`` (head -> foot), so attention maps straight back to slices for
the report notebooks. Padding MUST be masked out (e.g. scores set to -inf before the softmax), or a
volume's result would depend on which other volumes share its batch.
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


class Aggregator(nn.Module, ABC):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

    @abstractmethod
    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> AggregatorOutput:
        ...
