"""Stage 3 + 4 components against their definitions: ABMIL (Ilse et al. 2018, Eq. 7 and 9), the mean-pool
baseline, the linear head's prior, FeatureNorm, the registries, and the exact slice decomposition."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ct_model.aggregators.abmil import GatedABMIL  # noqa: E402
from ct_model.models.volume_classifier import FeatureNorm, VolumeClassifier  # noqa: E402
from ct_model.registry import AGGREGATORS, HEADS  # noqa: E402

torch.manual_seed(0)


def _abmil(**kw):
    m = GatedABMIL(in_dim=6, n_labels=3, hidden_dim=8, attn_dim=4, **{"dropout": 0.0, **kw})
    return m.eval()


def test_gated_attention_is_eq9_and_pooling_is_eq7():
    m = _abmil()
    for layer in (m.attn_V, m.attn_U, m.attn_w):
        torch.nn.init.zeros_(layer.bias)  # the paper's equation has no biases
    x = torch.randn(1, 7, 6)
    h = m.project(x)[0]
    V, U, w = m.attn_V.weight, m.attn_U.weight, m.attn_w.weight[0]
    s = torch.stack([w @ (torch.tanh(V @ hk) * torch.sigmoid(U @ hk)) for hk in h])
    a = torch.exp(s) / torch.exp(s).sum()
    out = m(x, torch.ones(1, 7, dtype=torch.bool))
    torch.testing.assert_close(out.attention[0], a)
    torch.testing.assert_close(out.pooled[0], (a[:, None] * h).sum(0))


@pytest.mark.parametrize("branches", ["single", "per_label"])
def test_padding_and_order_do_not_matter(branches):
    m = _abmil(branches=branches).requires_grad_(False)
    x = torch.randn(2, 6, 6)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    out = m(x, mask)
    alone = m(x[1:, :3], mask[1:, :3])
    torch.testing.assert_close(alone.pooled, out.pooled[1:])
    perm = torch.tensor([3, 0, 5, 1, 4, 2])
    torch.testing.assert_close(m(x[:1, perm], mask[:1, perm]).pooled, out.pooled[:1])
    att = out.attention if branches == "per_label" else out.attention[:, None]
    torch.testing.assert_close(att.sum(-1), torch.ones(att.shape[:2]))
    assert float(att[1, :, 3:].abs().max()) == 0.0


def test_per_label_branches_attend_independently():
    m = _abmil(branches="per_label")
    out = m(torch.randn(2, 5, 6), torch.ones(2, 5, dtype=torch.bool))
    assert out.pooled.shape == (2, 3, 8) and out.attention.shape == (2, 3, 5)
    assert not torch.allclose(out.attention[:, 0], out.attention[:, 1])


def test_gradients_reach_every_parameter():
    m = GatedABMIL(6, 3, hidden_dim=8, attn_dim=4).train()
    m(torch.randn(2, 5, 6), torch.ones(2, 5, dtype=torch.bool)).pooled.sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_mean_pool_is_the_masked_mean():
    m = AGGREGATORS.build({"type": "mean_pool", "hidden_dim": 8, "dropout": 0.0}, in_dim=6, n_labels=3).eval()
    x, mask = torch.randn(1, 5, 6), torch.tensor([[1, 1, 0, 1, 0]], dtype=torch.bool)
    out = m(x, mask)
    torch.testing.assert_close(out.pooled[0], m.project(x)[0, mask[0]].mean(0))


def test_registries_are_strict():
    with pytest.raises(ValueError, match="registered"):
        AGGREGATORS.build({"type": "nope"}, in_dim=4, n_labels=2)
    with pytest.raises(ValueError, match="unexpected keyword"):
        AGGREGATORS.build({"type": "abmil", "hiden_dim": 8}, in_dim=4, n_labels=2)
    with pytest.raises(ValueError, match="branches"):
        AGGREGATORS.build({"type": "abmil", "branches": "both"}, in_dim=4, n_labels=2)
    with pytest.raises(NotImplementedError, match="phase 3"):
        AGGREGATORS.build({"type": "query_mil"}, in_dim=4, n_labels=2)
    assert set(AGGREGATORS.names()) >= {"abmil", "mean_pool", "query_mil"} and "linear" in HEADS.names()


def test_head_prior_bias_predicts_train_prevalence():
    head = HEADS.build({"type": "linear"}, in_dim=8, n_labels=3)
    prev = [0.466, 0.25, 0.074]
    head.init_prior(prev)
    np.testing.assert_allclose(torch.sigmoid(head.linear.bias).detach().numpy(), prev, rtol=1e-5)


def test_feature_norm_matches_numpy():
    bags = [np.random.default_rng(i).normal(3, 2, size=(n, 6)) for i, n in enumerate((4, 9, 2))]
    fn = FeatureNorm(6).fit(bags)
    allx = np.concatenate(bags)
    np.testing.assert_allclose(fn.mean.numpy(), allx.mean(0), rtol=1e-5)
    np.testing.assert_allclose(fn.std.numpy(), allx.std(0), rtol=1e-5)


@pytest.mark.parametrize("spec", [{"type": "abmil"}, {"type": "abmil", "branches": "per_label"}, {"type": "mean_pool"}])
def test_contributions_sum_exactly_to_the_logits(spec):
    agg = AGGREGATORS.build({**spec, "hidden_dim": 8}, in_dim=6, n_labels=3)
    model = VolumeClassifier(FeatureNorm(6).fit([np.random.randn(10, 6)]), agg,
                             HEADS.build({"type": "linear", "dropout": 0.3}, in_dim=8, n_labels=3), ["a", "b", "c"]).eval()
    x, mask = torch.randn(2, 6, 6), torch.tensor([[1] * 6, [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    logits = model(x, mask).logits
    contrib, bias = model.contributions(x, mask)
    assert contrib.shape == (2, 3, 6) and float(contrib[1, :, 4:].abs().max()) == 0.0
    torch.testing.assert_close(contrib.sum(-1) + bias, logits)
    with pytest.raises(RuntimeError, match="eval mode"):
        model.train().contributions(x, mask)
