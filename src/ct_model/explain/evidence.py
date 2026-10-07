"""L1 evidence: which slices drive each label's prediction -- exact, from the trained model alone.

For an aggregator with linear pooling and the linear head (ABMIL, mean pool),

    logit_c = b_c + sum_k  a_{c,k} (w_c . h_k)

(see ct_model.models.volume_classifier), so ``contributions[c, k]`` is slice k's exact share of
label c's logit: positive slices push towards "present", negative ones away. No gradients, no
approximation; ``check_sum`` verifies the identity on every call.

Slice k is slice k of the PREPROCESSED volume (the HU cache / run_inference output): head -> foot at
1.5 mm, body-cropped. The crop box is not stored by preprocessing yet, so mapping back to the original
DICOM slice number is not possible today (docs/model/architecture.md, TODO).

This is evidence of what the MODEL used, not a validated localisation of the finding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models.volume_classifier import VolumeClassifier


@dataclass
class SliceEvidence:
    label_names: list[str]
    probs: np.ndarray          # (C,)
    logits: np.ndarray         # (C,)
    thresholds: np.ndarray     # (C,) chosen on val; NaN if a label could not be scored there
    contributions: np.ndarray  # (C, N): slice k's share of label c's logit
    bias: np.ndarray           # (C,)
    attention: np.ndarray      # (C, N) for per-label aggregators, (1, N) otherwise

    @property
    def predicted(self) -> np.ndarray:
        """(C,) bool: probability >= the label's val threshold (False where no threshold exists)."""
        with np.errstate(invalid="ignore"):
            return np.where(np.isnan(self.thresholds), False, self.probs >= self.thresholds)

    def top_slices(self, label: int | str, k: int = 5) -> np.ndarray:
        """Indices of the ``k`` slices with the largest (most positive) contribution to ``label``."""
        c = self.label_names.index(label) if isinstance(label, str) else label
        return np.argsort(-self.contributions[c], kind="stable")[:k]


@torch.no_grad()
def slice_evidence(model: VolumeClassifier, bag: np.ndarray, thresholds: np.ndarray | None = None,
                   device: str | torch.device = "cpu", check_sum: bool = True) -> SliceEvidence:
    """``bag``: one volume's (n_slices, D) embeddings from the store."""
    model = model.eval()
    x = torch.as_tensor(np.asarray(bag, dtype=np.float32), device=device)[None]
    mask = torch.ones(x.shape[:2], dtype=torch.bool, device=device)
    out = model(x, mask)
    contrib, bias = model.contributions(x, mask)
    contrib, bias, logits = contrib[0].cpu().numpy(), bias.cpu().numpy(), out.logits[0].cpu().numpy()
    if check_sum:
        err = np.abs(contrib.sum(1) + bias - logits).max()
        if err > 1e-3 * max(1.0, np.abs(logits).max()):
            raise RuntimeError(f"slice contributions do not add up to the logits (max error {err:.2e}) -- not exact here")
    attention = out.attention[0].cpu().numpy()
    names = model.label_names
    return SliceEvidence(
        label_names=names, probs=1 / (1 + np.exp(-logits)), logits=logits,
        thresholds=np.full(len(names), np.nan) if thresholds is None else np.asarray(thresholds, dtype=float),
        contributions=contrib, bias=bias, attention=attention if attention.ndim == 2 else attention[None],
    )
