"""Losses against references: torch's BCEWithLogitsLoss (incl. pos_weight) and an independent numpy
implementation of the ASL reference code (Alibaba-MIIL/ASL, AsymmetricLoss)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ct_model.registry import LOSSES  # noqa: E402
from ct_model.training.losses import train_pos_weight  # noqa: E402

LOGITS = torch.tensor([[2.0, -1.0, 0.3], [-3.0, 0.5, 4.0], [0.0, 1.5, -2.5]])
TARGETS = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
ALL = torch.ones_like(TARGETS, dtype=torch.bool)
TRAIN = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=float)


def asl_reference_numpy(x, y, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
    """The reference AsymmetricLoss.forward, re-written in numpy (returns the SUM, like the reference)."""
    xs_pos = 1 / (1 + np.exp(-x))
    xs_neg = np.minimum(1 - xs_pos + clip, 1) if clip > 0 else 1 - xs_pos
    loss = y * np.log(np.maximum(xs_pos, eps)) + (1 - y) * np.log(np.maximum(xs_neg, eps))
    if gamma_neg > 0 or gamma_pos > 0:
        pt = xs_pos * y + xs_neg * (1 - y)
        loss = loss * np.power(1 - pt, gamma_pos * y + gamma_neg * (1 - y))
    return -loss.sum()


def test_bce_equals_torch():
    ours = LOSSES.build({"type": "bce"}, train_labels=TRAIN)(LOGITS, TARGETS, ALL)
    torch.testing.assert_close(ours, torch.nn.BCEWithLogitsLoss()(LOGITS, TARGETS))


def test_weighted_bce_equals_torch_pos_weight():
    pw = train_pos_weight(TRAIN)
    torch.testing.assert_close(pw, torch.tensor([3.0, 3.0, 1 / 3]))  # negatives / positives per label
    ours = LOSSES.build({"type": "weighted_bce"}, train_labels=TRAIN)(LOGITS, TARGETS, ALL)
    torch.testing.assert_close(ours, torch.nn.BCEWithLogitsLoss(pos_weight=pw)(LOGITS, TARGETS))


def test_asl_equals_reference_formula():
    ours = LOSSES.build({"type": "asl"}, train_labels=TRAIN)(LOGITS, TARGETS, ALL)
    ref = asl_reference_numpy(LOGITS.numpy().astype(np.float64), TARGETS.numpy()) / TARGETS.numel()
    assert float(ours) == pytest.approx(ref, rel=1e-5)
    custom = LOSSES.build({"type": "asl", "gamma_neg": 2, "gamma_pos": 0, "clip": 0.1}, train_labels=TRAIN)
    ref = asl_reference_numpy(LOGITS.numpy().astype(np.float64), TARGETS.numpy(), 2, 0, 0.1) / TARGETS.numel()
    assert float(custom(LOGITS, TARGETS, ALL)) == pytest.approx(ref, rel=1e-5)


def test_asl_without_focusing_or_clipping_is_bce():
    asl = LOSSES.build({"type": "asl", "gamma_neg": 0, "gamma_pos": 0, "clip": 0}, train_labels=TRAIN)
    torch.testing.assert_close(asl(LOGITS, TARGETS, ALL), torch.nn.BCEWithLogitsLoss()(LOGITS, TARGETS))


def test_asl_focusing_weight_carries_no_gradient():
    """The reference computes the focusing weight with gradients disabled."""
    x = LOGITS.clone().requires_grad_(True)
    LOSSES.build({"type": "asl"}, train_labels=TRAIN)(x, TARGETS, ALL).backward()
    p = torch.sigmoid(LOGITS)
    xs_neg = (1 - p + 0.05).clamp(max=1)
    pt = p * TARGETS + xs_neg * (1 - TARGETS)
    w = (1 - pt) ** (1.0 * TARGETS + 4.0 * (1 - TARGETS))
    # d/dx of -[y log p + (1-y) log xs_neg] * w with w constant
    dneg = torch.where(1 - p + 0.05 < 1, p * (1 - p) / xs_neg, torch.zeros_like(p))
    expected = (-(TARGETS * (1 - p)) + (1 - TARGETS) * dneg) * w / TARGETS.numel()
    torch.testing.assert_close(x.grad, expected)


@pytest.mark.parametrize("kind", ["bce", "weighted_bce", "asl"])
def test_masked_labels_contribute_nothing(kind):
    loss = LOSSES.build({"type": kind}, train_labels=TRAIN)
    mask = ALL.clone()
    mask[0, 1] = False
    changed = TARGETS.clone()
    changed[0, 1] = 1 - changed[0, 1]
    assert float(loss(LOGITS, TARGETS, mask)) == pytest.approx(float(loss(LOGITS, changed, mask)))


def test_pos_weight_needs_positives():
    with pytest.raises(ValueError, match="no positive"):
        train_pos_weight(np.zeros((3, 2)))
