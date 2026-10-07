"""TODO (stage 3, phase 3): query-based MIL / QGMIL -- replaces ABMIL once phase 1 has a baseline.

Intended design: Q learnable queries (e.g. one per abnormality label, Q = 18) cross-attend over the
slice embeddings (nn.MultiheadAttention with ``key_padding_mask = ~mask``), optionally after a small
transformer encoder over slices with a 1D positional encoding of slice position (head -> foot).
Output AggregatorOutput(pooled=(B, Q, out_dim), attention=(B, Q, N)), so each label gets its own
slice attention map; the head then maps each query embedding to its own label's logit.

Compare against ABMIL on the same embedding store, splits and seeds -- only the experiment config changes.
"""
from __future__ import annotations

from .base import Aggregator, AggregatorOutput


class QueryMIL(Aggregator):
    def __init__(self, in_dim: int, n_queries: int, hidden_dim: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__(in_dim, hidden_dim)
        raise NotImplementedError("QueryMIL / QGMIL is a TODO for phase 3 -- see this module's docstring")

    def forward(self, bags, mask) -> AggregatorOutput:  # pragma: no cover - TODO
        raise NotImplementedError
