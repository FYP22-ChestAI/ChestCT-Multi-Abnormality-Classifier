"""Stage 3: a bag of slice embeddings -> one volume embedding (+ attention over slices).

Phase 1: ABMIL (abmil.py, TODO). Phase 3: query-based / QGMIL (query_mil.py, TODO).
Every aggregator implements ``Aggregator`` (base.py), so heads, training and reports never depend on which one.
"""
from .base import Aggregator, AggregatorOutput

__all__ = ["Aggregator", "AggregatorOutput"]
