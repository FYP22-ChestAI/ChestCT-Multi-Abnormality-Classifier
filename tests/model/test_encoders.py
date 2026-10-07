"""HU transforms and the timm backbone wrapper (tiny random ViT, no download)."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ct_model.config import InputConfig, load_encoder_config, parse_encoder_config  # noqa: E402
from ct_model.encoders import build_encoder  # noqa: E402
from ct_model.encoders.timm_vit import load_checkpoint  # noqa: E402
from ct_model.encoders.transforms import InputTransform  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def dale_card_transform(hu: np.ndarray) -> np.ndarray:
    """CTInferenceTransform, verbatim from the DALE-CT-2S model card."""
    clip_min, clip_max, mean_hu, std_hu = -997.0, 888.0, -142.39, 360.97
    range_val = clip_max - clip_min
    norm_mean, norm_std = (mean_hu - clip_min) / range_val, std_hu / range_val
    v = (np.clip(hu.astype(np.float64), clip_min, clip_max) - clip_min) / range_val
    return (v - norm_mean) / norm_std


def test_dale_transform_equals_model_card():
    cfg = load_encoder_config("dale_ct_2s", REPO / "configs" / "model" / "encoders")
    hu = np.random.default_rng(0).integers(-3000, 3000, size=(3, 224, 224)).astype(np.int16)
    ours = InputTransform(cfg.input)(torch.from_numpy(hu)).numpy()
    assert ours.shape == (3, 1, 224, 224)
    np.testing.assert_allclose(ours[:, 0], dale_card_transform(hu), rtol=0, atol=1e-5)


def test_multi_window_and_resize():
    cfg = InputConfig(transform="multi_window", windows=[(-1000, 400), (-150, 250)], channel_mean=[0.5, 0.5],
                      channel_std=[0.5, 0.5], size_hw=(48, 48))
    hu = torch.tensor([[-2000, -1000], [250, 3000]], dtype=torch.int16).repeat(2, 12, 12)  # (2, 24, 24)
    x = InputTransform(cfg)(hu)
    assert x.shape == (2, 2, 48, 48)
    assert x.min() >= -1.0 - 1e-6 and x.max() <= 1.0 + 1e-6  # [0, 1] windows -> [-1, 1] after (x - .5) / .5


def _tiny(pooling="cls", size=32, **over):
    raw = {
        "name": "tiny", "arch": "vit_tiny_patch16_224",
        "model_args": {"img_size": 64, "in_chans": 1, "num_classes": 0, "dynamic_img_size": True},
        "weights": {"source": "none"}, "pooling": pooling,
        "input": {"transform": "clip_zscore", "clip_hu": [-997, 888], "mean_hu": -142.39, "std_hu": 360.97,
                  "size_hw": [size, size]},
    }
    raw.update(over)
    return parse_encoder_config(raw)


def test_cls_pooling_equals_model_forward_and_is_frozen():
    enc = build_encoder(_tiny())
    assert enc.embed_dim == 192 and not any(p.requires_grad for p in enc.parameters()) and not enc.training
    hu = torch.randint(-1000, 1000, (5, 32, 32), dtype=torch.int16)
    with torch.inference_mode():
        out = enc(hu)
        ref = enc.model(enc.input_transform(hu))  # what the model card calls the "global feature"
    assert out.shape == (5, 192)
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("pooling,dim", [("mean_patch", 192), ("cls_mean", 384)])
def test_other_poolings(pooling, dim):
    enc = build_encoder(_tiny(pooling))
    with torch.inference_mode():
        assert enc(torch.zeros(2, 32, 32, dtype=torch.int16)).shape == (2, dim)


def test_input_must_fit_patch_size_channels_and_embed_dim():
    with pytest.raises(ValueError, match="multiple of"):
        build_encoder(_tiny(size=30))
    with pytest.raises(ValueError, match="input channel"):
        build_encoder(_tiny(input={"transform": "multi_window", "windows": [[-1000, 400]] * 3, "size_hw": [32, 32]}))
    with pytest.raises(ValueError, match="embed_dim"):
        build_encoder(_tiny(embed_dim=1024))


def test_lora_is_not_silently_ignored():
    with pytest.raises(NotImplementedError, match="LoRA"):
        build_encoder(_tiny(lora={"r": 8}))


def test_checkpoint_loading_is_strict(tmp_path):
    from safetensors.torch import save_file

    src = build_encoder(_tiny()).model
    state = {k: v.contiguous() for k, v in src.state_dict().items()}
    save_file(state, str(tmp_path / "ok.safetensors"))
    dst = build_encoder(_tiny()).model
    load_checkpoint(dst, tmp_path / "ok.safetensors")
    torch.testing.assert_close(dst.pos_embed, src.pos_embed)

    save_file({**state, "aux_head.weight": torch.zeros(3)}, str(tmp_path / "extra.safetensors"))
    with pytest.raises(RuntimeError, match="aux_head.weight"):
        load_checkpoint(dst, tmp_path / "extra.safetensors")
    report = load_checkpoint(dst, tmp_path / "extra.safetensors", ignore_prefixes=["aux_head."])
    assert report["ignored_unexpected"] == ["aux_head.weight"]

    del state["cls_token"]
    save_file(state, str(tmp_path / "missing.safetensors"))
    with pytest.raises(RuntimeError, match="cls_token"):
        load_checkpoint(dst, tmp_path / "missing.safetensors")
