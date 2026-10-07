"""Multi-label losses. Every loss takes ``(logits (B, C), targets (B, C), label_mask (B, C) bool)`` and
returns the MEAN over unmasked entries, so missing labels contribute nothing and the loss scale
(hence the learning rate's meaning) does not depend on the batch size or the loss type.

    bce           binary cross-entropy with logits
    weighted_bce  the same with pos_weight_c = negatives_c / positives_c counted on the TRAIN split
                  (torch.nn.BCEWithLogitsLoss's pos_weight: up-weights the positive term of rare labels)
    asl           Asymmetric Loss (Ridnik et al., ICCV 2021), as implemented in the authors' reference
                  code (github.com/Alibaba-MIIL/ASL, src/loss_functions/losses.py, AsymmetricLoss):
                    p = sigmoid(x);  p_neg = min(1 - p + clip, 1)              (probability shifting)
                    loss = -[ y log(max(p, eps)) + (1 - y) log(max(p_neg, eps)) ] * (1 - pt)^gamma
                    pt = p y + p_neg (1 - y);  gamma = gamma_pos y + gamma_neg (1 - y)
                  with the reference defaults gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, and the focusing
                  weight treated as a constant (the reference computes it with gradients disabled). The
                  reference returns the SUM; here it is divided by the number of unmasked entries.

Which loss handles CT-RATE's imbalance best is decided by experiment (docs/model/README.md), not here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..registry import LOSSES


def _masked_mean(per_entry: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(per_entry.dtype)
    return (per_entry * mask).sum() / mask.sum().clamp(min=1.0)


class BCELoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None, persistent=False)

    def forward(self, logits, targets, mask):
        per_entry = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight, reduction="none")
        return _masked_mean(per_entry, mask)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, targets, mask):
        p = torch.sigmoid(logits)
        p_neg = 1 - p
        if self.clip is not None and self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1)
        loss = targets * torch.log(p.clamp(min=self.eps)) + (1 - targets) * torch.log(p_neg.clamp(min=self.eps))
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            with torch.no_grad():
                pt = p * targets + p_neg * (1 - targets)
                weight = torch.pow(1 - pt, self.gamma_pos * targets + self.gamma_neg * (1 - targets))
            loss = loss * weight
        return _masked_mean(-loss, mask)


def train_pos_weight(train_labels) -> torch.Tensor:
    """negatives / positives per label, from the TRAIN labels (n_volumes, C), NaN = missing."""
    y = torch.as_tensor(train_labels, dtype=torch.float64)
    known = ~torch.isnan(y)
    pos = torch.where(known, y, torch.zeros_like(y)).sum(0)
    neg = known.sum(0) - pos
    if (pos == 0).any():
        raise ValueError("a label has no positive training volume -- pos_weight is undefined; drop the label or use bce")
    return (neg / pos).float()


@LOSSES.register("bce")
def build_bce(train_labels) -> BCELoss:
    return BCELoss()


@LOSSES.register("weighted_bce")
def build_weighted_bce(train_labels) -> BCELoss:
    return BCELoss(pos_weight=train_pos_weight(train_labels))


@LOSSES.register("asl")
def build_asl(train_labels, **params) -> AsymmetricLoss:
    return AsymmetricLoss(**params)
