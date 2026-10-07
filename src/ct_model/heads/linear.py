"""TODO (stage 4, phase 1): the multi-label classifier head.

Intended design: ``Dropout -> Linear(in_dim, n_labels)``, with the bias initialised to log(p / (1 - p)),
p = each label's TRAIN prevalence (a faster, more stable start for rare labels). For a query-based
aggregator ((B, Q, in_dim) with Q == n_labels), one linear map per query instead (an einsum over Q).
"""
from __future__ import annotations

from .base import Head


class LinearHead(Head):
    def __init__(self, in_dim: int, n_labels: int, dropout: float = 0.0, prior: list[float] | None = None):
        super().__init__(in_dim, n_labels)
        raise NotImplementedError("LinearHead is a TODO for stage 4 -- see this module's docstring")

    def forward(self, pooled):  # pragma: no cover - TODO
        raise NotImplementedError
