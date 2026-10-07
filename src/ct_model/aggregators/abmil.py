"""Gated attention-based MIL (Ilse, Tomczak & Welling, "Attention-based Deep Multiple Instance
Learning", ICML 2018, arXiv:1802.04712).

From the paper:

    (7)  z   = sum_k a_k h_k
    (9)  a_k = exp{w^T (tanh(V h_k^T) * sigm(U h_k^T))} / sum_j exp{w^T (tanh(V h_j^T) * sigm(U h_j^T))}
         with w in R^{L x 1}, V, U in R^{L x M}

Here the per-slice embedding is ``h_k = Dropout(ReLU(Linear(x_k)))`` (the "f" of the paper,
D -> ``hidden_dim``), and the paper's L (attention size) is ``attn_dim``. V and U are nn.Linear
layers, so they carry a bias the paper's equation omits (as common implementations do). Padded slices
get exactly zero weight (masked softmax).

``branches``:
  * ``single``    -- the paper: one attention over slices, one volume embedding, shared by all labels.
  * ``per_label`` -- ``w`` has one column per label, so every label has its own attention over slices
                     and its own pooled embedding (B, C, L). More interpretable evidence per label;
                     whether it also predicts better is an experiment, not an assumption.
"""
from __future__ import annotations

import torch
from torch import nn

from ..registry import AGGREGATORS
from .base import Aggregator, AggregatorOutput, masked_softmax

_BRANCHES = ("single", "per_label")


class GatedABMIL(Aggregator):
    linear_pooling = True

    def __init__(self, in_dim: int, n_labels: int, hidden_dim: int = 256, attn_dim: int = 128,
                 dropout: float = 0.25, branches: str = "single"):
        if branches not in _BRANCHES:
            raise ValueError(f"branches must be one of {list(_BRANCHES)}, got {branches!r}")
        super().__init__(in_dim, hidden_dim)
        self.per_label = branches == "per_label"
        self.project = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, n_labels if self.per_label else 1)

    def scores(self, h: torch.Tensor) -> torch.Tensor:
        """Eq. 9 before the softmax: (B, N, L) -> (B, branches, N)."""
        return self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))).transpose(1, 2)

    def forward(self, bags: torch.Tensor, mask: torch.Tensor) -> AggregatorOutput:
        h = self.project(bags)                       # (B, N, L)
        a = masked_softmax(self.scores(h), mask)     # (B, branches, N)
        pooled = a @ h                               # Eq. 7: (B, branches, L)
        if not self.per_label:
            return AggregatorOutput(pooled=pooled[:, 0], attention=a[:, 0], instances=h)
        return AggregatorOutput(pooled=pooled, attention=a, instances=h)


@AGGREGATORS.register("abmil")
def build_abmil(in_dim: int, n_labels: int, **params) -> GatedABMIL:
    return GatedABMIL(in_dim, n_labels, **params)
