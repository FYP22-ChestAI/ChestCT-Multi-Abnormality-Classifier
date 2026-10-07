"""Stages 3 + 4 composed: slice embeddings -> standardised -> aggregator -> head -> 18 logits.

    VolumeClassifier(feature_norm, aggregator, head, label_names)
        forward(bags (B, N, D), mask (B, N))  ->  ClassifierOutput(logits (B, C), attention)
        contributions(bags, mask)             ->  (B, C, N) per-slice share of every logit, + bias (C)

``FeatureNorm`` standardises every embedding dimension with a mean / std fitted on the TRAIN split
only. They are buffers, so they travel inside the checkpoint and inference needs nothing else.

Exact slice evidence. With an aggregator whose ``pooled = sum_k a_{c,k} h_k`` (``linear_pooling``) and
the linear head:

    logit_c = b_c + w_c . sum_k a_{c,k} h_k = b_c + sum_k  a_{c,k} (w_c . h_k)

so ``a_{c,k} (w_c . h_k)`` is slice k's exact contribution to label c (a_{c,k} = a_k for a
single-branch aggregator). Exact in eval mode (dropout off); test_mil_models checks the sum.

Phase 2 (LoRA) will feed HU slices through the encoder into the same aggregator and head; this class
then gains the encoder in front (see docs/model/architecture.md). Phase 1 never needs it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..aggregators.base import Aggregator
from ..heads.linear import LinearHead


class FeatureNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, bags) -> "FeatureNorm":
        """Mean / std over every slice of every bag (an iterable of (n_slices, D) arrays), in float64."""
        total, sq, count = None, None, 0
        for bag in bags:
            b = np.asarray(bag, dtype=np.float64)
            total = b.sum(0) if total is None else total + b.sum(0)
            sq = (b * b).sum(0) if sq is None else sq + (b * b).sum(0)
            count += b.shape[0]
        if not count:
            raise ValueError("cannot fit FeatureNorm on no slices")
        mean = total / count
        std = np.sqrt(np.maximum(sq / count - mean * mean, 0.0))
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.std.copy_(torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


@dataclass
class ClassifierOutput:
    logits: torch.Tensor
    attention: torch.Tensor | None


class VolumeClassifier(nn.Module):
    def __init__(self, feature_norm: FeatureNorm, aggregator: Aggregator, head: nn.Module, label_names: list[str]):
        super().__init__()
        if head.n_labels != len(label_names):
            raise ValueError(f"head has {head.n_labels} outputs but there are {len(label_names)} labels")
        self.feature_norm = feature_norm
        self.aggregator = aggregator
        self.head = head
        self.label_names = list(label_names)

    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> ClassifierOutput:
        out = self.aggregator(self.feature_norm(bags), mask)
        return ClassifierOutput(logits=self.head(out.pooled), attention=out.attention)

    @torch.no_grad()
    def contributions(self, bags: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """((B, C, N) per-slice contributions, (C,) bias); padding contributes 0. Eval mode only."""
        if self.training:
            raise RuntimeError("contributions are exact only in eval mode (dropout off) -- call model.eval()")
        if not self.aggregator.linear_pooling or not isinstance(self.head, LinearHead):
            raise NotImplementedError(
                "exact slice contributions need an aggregator with linear_pooling and the linear head; "
                f"got {type(self.aggregator).__name__} + {type(self.head).__name__}"
            )
        out = self.aggregator(self.feature_norm(bags), mask)
        per_slice = out.instances @ self.head.linear.weight.T          # (B, N, C): w_c . h_k
        a = out.attention if out.attention.ndim == 3 else out.attention.unsqueeze(1)  # (B, C or 1, N)
        return a * per_slice.transpose(1, 2), self.head.linear.bias.detach().clone()
