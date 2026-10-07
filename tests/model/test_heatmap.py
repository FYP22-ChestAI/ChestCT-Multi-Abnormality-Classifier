"""L2 in-slice heatmaps with a tiny random ViT (no download): the Grad-CAM arithmetic against an
independent autograd computation, the re-encode guard, and the two diagnostics."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ct_model.config import parse_encoder_config  # noqa: E402
from ct_model.encoders import build_encoder  # noqa: E402
from ct_model.explain.heatmap import (  # noqa: E402
    EvidenceMismatch, deletion_check, randomization_check, slice_heatmap,
)
from ct_model.models.volume_classifier import FeatureNorm, VolumeClassifier  # noqa: E402
from ct_model.registry import AGGREGATORS, HEADS  # noqa: E402

LABELS = ["a", "b"]


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    enc = build_encoder(parse_encoder_config({
        "name": "tiny", "arch": "vit_tiny_patch16_224", "weights": {"source": "none"},
        "model_args": {"img_size": 64, "in_chans": 1, "num_classes": 0, "dynamic_img_size": True},
        "input": {"transform": "clip_zscore", "clip_hu": [-997, 888], "mean_hu": -142.39, "std_hu": 360.97,
                  "size_hw": [32, 48]},
    }))
    rng = np.random.default_rng(0)
    hu = rng.integers(-1000, 600, size=(6, 32, 48)).astype(np.int16)
    with torch.no_grad():
        bag = enc(torch.from_numpy(hu)).numpy()
    model = VolumeClassifier(FeatureNorm(192).fit([bag]), AGGREGATORS.build({"type": "abmil", "hidden_dim": 16}, in_dim=192, n_labels=2),
                             HEADS.build({"type": "linear"}, in_dim=16, n_labels=2), LABELS).eval()
    return enc, model, bag, hu


def test_heatmap_shape_range_and_reencode(setup):
    enc, model, bag, hu = setup
    h = slice_heatmap(model, enc, bag, hu[2], 2, "b")
    assert h.heatmap.shape == (32, 48) and h.patch_cam.shape == (2, 3)  # 32/16 x 48/16 patches
    assert h.heatmap.min() >= 0 and h.heatmap.max() <= 1 + 1e-6
    assert h.reencode_cosine > 0.9999
    with torch.no_grad():
        full = model(torch.from_numpy(bag)[None], torch.ones(1, 6, dtype=torch.bool)).logits[0, 1]
    assert h.logit == pytest.approx(float(full), abs=1e-4)
    assert all(p.grad is None for p in model.parameters())  # no gradients left behind


def test_gradcam_matches_independent_autograd(setup):
    enc, model, bag, hu = setup
    h = slice_heatmap(model, enc, bag, hu[4], 4, "a")
    vit = enc.model
    x = enc.input_transform(torch.from_numpy(hu[4:5]))
    tokens = {}
    handle = vit.blocks[-1].register_forward_hook(lambda m, i, o: tokens.setdefault("t", o))
    emb = enc.embed(x.requires_grad_(True))
    handle.remove()
    full = torch.cat([torch.from_numpy(bag[:4]), emb, torch.from_numpy(bag[5:])])[None]
    logit = model(full, torch.ones(1, 6, dtype=torch.bool)).logits[0, 0]
    (grad,) = torch.autograd.grad(logit, tokens["t"])
    t, g = tokens["t"][0, 1:].detach(), grad[0, 1:]
    cam = torch.relu((t * g.mean(0)).sum(-1)).reshape(2, 3).numpy()
    np.testing.assert_allclose(h.patch_cam, cam, rtol=1e-4, atol=1e-6)
    model.zero_grad(set_to_none=True)


def test_mismatched_embedding_is_refused(setup):
    enc, model, bag, hu = setup
    with pytest.raises(EvidenceMismatch, match="cosine"):
        slice_heatmap(model, enc, bag, hu[3], 2, "a")  # slice 3's pixels against slice 2's stored embedding


def test_diagnostics_run_and_restore_the_model(setup):
    enc, model, bag, hu = setup
    before = {k: v.clone() for k, v in model.state_dict().items()}
    r = randomization_check(model, enc, bag, hu[1], 1, "a")
    assert np.isnan(r) or -1 <= r <= 1
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())  # head restored
    h = slice_heatmap(model, enc, bag, hu[1], 1, "a")
    d = deletion_check(model, enc, bag, hu[1], 1, "a", h.patch_cam, fraction=0.34, n_random=3)
    assert d["n_patches_deleted"] == 2 and d["logit"] == pytest.approx(h.logit, abs=1e-4)
    assert {"drop_top", "drop_random_mean"} <= set(d)
