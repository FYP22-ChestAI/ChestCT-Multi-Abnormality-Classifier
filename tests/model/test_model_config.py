"""The model YAML files: strict loading, and a fingerprint that tracks every setting."""
from pathlib import Path

import pytest

from ct_model.config import (
    load_encoder_config, load_stage2_config, parse_encoder_config,
)

REPO = Path(__file__).resolve().parents[2]
ENCODERS = REPO / "configs" / "model" / "encoders"


def test_repo_configs_load():
    stage2 = load_stage2_config(REPO / "configs" / "model" / "encode.yaml")
    assert stage2.encoder == "dale_ct_2s"
    assert stage2.selection.splits == ("train", "val", "test")
    for path in ENCODERS.glob("*.yaml"):
        cfg = load_encoder_config(str(path))
        assert cfg.name == path.stem


def test_dale_config_matches_model_card():
    cfg = load_encoder_config("dale_ct_2s", ENCODERS)
    assert cfg.arch == "vit_large_patch14_dinov2"
    assert cfg.model_args == {"patch_size": 16, "img_size": 512, "in_chans": 1, "num_classes": 0, "dynamic_img_size": True}
    assert cfg.input.clip_hu == (-997.0, 888.0) and cfg.input.mean_hu == -142.39 and cfg.input.std_hu == 360.97
    assert cfg.input.channels == 1 and cfg.embed_dim == 1024
    assert cfg.weights.revision and len(cfg.weights.revision) == 40  # pinned commit, not a branch


def test_encoder_name_lookup_lists_available(tmp_path):
    with pytest.raises(FileNotFoundError, match="dale_ct_2s"):
        load_encoder_config("nope", ENCODERS)


def _raw(**over):
    raw = {"name": "x", "arch": "a", "input": {"transform": "clip_zscore", "clip_hu": [-1000, 1000], "mean_hu": 0, "std_hu": 1}}
    raw.update(over)
    return raw


def test_unknown_keys_and_bad_values_raise():
    with pytest.raises(ValueError, match="unexpected keyword"):
        parse_encoder_config(_raw(poolin="cls"))
    with pytest.raises(ValueError, match="pooling"):
        parse_encoder_config(_raw(pooling="max"))
    with pytest.raises(ValueError, match="clip_zscore needs"):
        parse_encoder_config(_raw(input={"transform": "clip_zscore"}))
    with pytest.raises(ValueError, match="one value per window"):
        parse_encoder_config(_raw(input={"transform": "multi_window", "windows": [[0, 1], [2, 3]], "channel_mean": [0.5]}))


def test_fingerprint_is_stable_and_tracks_settings():
    a, b = parse_encoder_config(_raw()), parse_encoder_config(_raw())
    assert a.fingerprint() == b.fingerprint()
    assert a.store_name == f"x-{a.fingerprint()[:8]}"
    changed = parse_encoder_config(_raw(input={**_raw()["input"], "size_hw": [256, 256]}))
    assert changed.fingerprint() != a.fingerprint()
    assert parse_encoder_config(_raw(pooling="mean_patch")).fingerprint() != a.fingerprint()
