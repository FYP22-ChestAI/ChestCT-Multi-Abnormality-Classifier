"""Stage 3: a bag of slice embeddings -> one volume embedding (+ attention over slices).

Built by ``aggregator.type`` in an experiment config (ct_model.registry.AGGREGATORS):
    abmil       gated attention MIL (Ilse et al. 2018), branches single | per_label   (abmil.py)
    mean_pool   masked mean -- the baseline                                          (mean_pool.py)
    query_mil   TODO, phase 3                                                          (query_mil.py)
"""
from .base import Aggregator, AggregatorOutput, masked_softmax

__all__ = ["Aggregator", "AggregatorOutput", "masked_softmax"]
