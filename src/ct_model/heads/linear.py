"""Stage 4: the multi-label linear head, ``logit_c = w_c . z + b_c``.

* ``pooled`` (B, L) from a single-branch aggregator: one shared volume embedding -> C logits.
* ``pooled`` (B, C, L) from a per-label aggregator: label c uses its own embedding z_c.
Both use the same (C, L) weight, so ``logit_c = w_c . z_c + b_c`` in either case.

``init_prior(prevalence)`` sets b_c = log(p_c / (1 - p_c)), so the untrained model already predicts
each label's TRAIN prevalence (sigmoid(b_c) = p_c) instead of 0.5 -- a calmer start for rare labels.
Being linear is what makes a logit an exact sum of per-slice contributions (see VolumeClassifier).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..registry import HEADS
from .base import Head


class LinearHead(Head):
    def __init__(self, in_dim: int, n_labels: int, dropout: float = 0.0):
        super().__init__(in_dim, n_labels)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, n_labels)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        z = self.dropout(pooled)
        if z.ndim == 2:
            return self.linear(z)
        if z.ndim == 3 and z.shape[1] == self.n_labels:
            return (z * self.linear.weight).sum(dim=-1) + self.linear.bias
        raise ValueError(f"expected pooled (B, {self.in_dim}) or (B, {self.n_labels}, {self.in_dim}), got {tuple(z.shape)}")

    @torch.no_grad()
    def init_prior(self, prevalence) -> None:
        p = np.clip(np.asarray(prevalence, dtype=np.float64), 1e-4, 1 - 1e-4)
        self.linear.bias.copy_(torch.as_tensor(np.log(p / (1 - p)), dtype=self.linear.bias.dtype))


@HEADS.register("linear")
def build_linear(in_dim: int, n_labels: int, **params) -> LinearHead:
    return LinearHead(in_dim, n_labels, **params)
