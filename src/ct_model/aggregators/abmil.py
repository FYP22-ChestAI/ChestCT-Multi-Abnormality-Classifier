"""TODO (stage 3, phase 1 -- next team member): gated attention-based MIL (Ilse et al., ICML 2018).

Intended design:

    h      = Dropout(LayerNorm(Linear(in_dim -> hidden_dim)(x)))        projection of the 1024-d slice embeddings
    s_n    = w^T ( tanh(V h_n) * sigmoid(U h_n) )                       gated attention score per slice
    a      = softmax_n(s), with s_n = -inf where mask is False          padding never gets weight
    pooled = sum_n a_n h_n                                              (B, hidden_dim)

Return AggregatorOutput(pooled, attention=a). Unit-test that (1) extra padding does not change a bag's
output, (2) attention sums to 1 over real slices, (3) shuffling slices does not change ``pooled``.
Hyper-parameters come from configs/model/experiments/*.yaml (aggregator.hidden_dim, aggregator.dropout).
"""
from __future__ import annotations

from .base import Aggregator, AggregatorOutput


class GatedABMIL(Aggregator):
    def __init__(self, in_dim: int, hidden_dim: int = 256, attn_dim: int = 128, dropout: float = 0.25):
        super().__init__(in_dim, hidden_dim)
        raise NotImplementedError("GatedABMIL is a TODO for stage 3 -- see the design in this module's docstring")

    def forward(self, bags, mask) -> AggregatorOutput:  # pragma: no cover - TODO
        raise NotImplementedError
