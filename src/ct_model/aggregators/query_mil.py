"""TODO (stage 3, phase 3): query-based MIL / QGMIL -- to be compared against ABMIL once phase 1 has
its baseline numbers.

Intended design: Q learnable queries (e.g. one per abnormality label, Q = 18) cross-attend over the
slice embeddings (nn.MultiheadAttention with ``key_padding_mask = ~mask``), optionally after a small
transformer encoder over slices with a 1D positional encoding of the slice position (head -> foot).
Output AggregatorOutput(pooled=(B, Q, out_dim), attention=(B, Q, N)) with ``per_label = True``, so the
linear head maps each query embedding to its own label's logit. If anything transforms the pooled
vectors after attention, set ``linear_pooling = False`` (slice evidence then needs a gradient method).

Compare against ABMIL on the same embedding store, splits and seeds: only the experiment config changes
(``aggregator: {type: query_mil, ...}``), and outputs/experiments/index.csv keeps both comparable.
"""
from __future__ import annotations

from ..registry import AGGREGATORS


@AGGREGATORS.register("query_mil")
def build_query_mil(in_dim: int, n_labels: int, **params):
    raise NotImplementedError("query_mil (QGMIL) is a TODO for phase 3 -- see ct_model/aggregators/query_mil.py")
