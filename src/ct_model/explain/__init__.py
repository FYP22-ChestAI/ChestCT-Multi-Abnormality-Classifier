"""Optional evidence for a trained classifier's predictions (nothing in training depends on it).

    evidence.py  L1 -- which slices: exact per-slice contributions to every label's logit
    heatmap.py   L2 -- where in a slice: Grad-CAM on the encoder's patch tokens, with sanity checks
    volume.py    everything for one volume of a finished run, used by scripts/model/explain_volume.py
                 and notebooks/evidence_viewer.ipynb

Evidence shows what the MODEL relied on. It is not a validated localisation of a finding.
"""
from .evidence import SliceEvidence, slice_evidence

__all__ = ["SliceEvidence", "slice_evidence"]
