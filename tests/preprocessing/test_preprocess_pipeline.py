"""End-to-end test of Phase 1 (load -> spacing -> crop -> resize -> save),
using the synthetic NIfTI fixture instead of a real CT-RATE download."""
import json

import numpy as np

from ct_preprocessing.preprocess import (
    PreprocessConfig, config_fingerprint, is_cache_fresh, load_cached_stats, preprocess_one,
)


def test_preprocess_one_produces_the_expected_npy(tmp_path, synthetic_nifti):
    path, _, _ = synthetic_nifti
    cfg = PreprocessConfig(
        target_spacing_zyx=(1.5, 0.75, 0.75),
        target_size_hw=(224, 224),
        crop_margin=4,
        crop_threshold_hu=-500.0,
        resize_mode="stretch",
    )

    result = preprocess_one(path, out_dir=tmp_path, cfg=cfg, volume_id="synthetic_1_a_1")

    assert result.ok, result.error
    assert result.out_shape == (result.n_slices, 224, 224)

    out = np.load(result.out_path)
    assert out.dtype == np.int16
    assert out.shape[1:] == (224, 224)
    # saved UNwindowed: real HU range should still be present, not [0, 1]
    assert out.min() < -500
    assert out.max() > 0


def test_preprocess_one_records_a_useful_error_for_a_bad_file(tmp_path):
    bad_path = tmp_path / "not_a_real_scan.nii.gz"
    bad_path.write_bytes(b"not a nifti file")

    result = preprocess_one(bad_path, out_dir=tmp_path, cfg=PreprocessConfig(), volume_id="bad_1")

    assert not result.ok
    assert result.error


def test_config_fingerprint_changes_when_settings_change():
    a = PreprocessConfig(target_size_hw=(224, 224))
    b = PreprocessConfig(target_size_hw=(112, 112))
    assert config_fingerprint(a) != config_fingerprint(b)
    assert config_fingerprint(a) == config_fingerprint(PreprocessConfig(target_size_hw=(224, 224)))


def test_preprocess_one_writes_a_fingerprint_sidecar(tmp_path, synthetic_nifti):
    path, _, _ = synthetic_nifti
    cfg = PreprocessConfig(target_size_hw=(224, 224))

    result = preprocess_one(path, out_dir=tmp_path, cfg=cfg, volume_id="synthetic_1_a_1")
    assert result.ok

    meta_path = tmp_path / "synthetic_1_a_1.meta.json"
    assert meta_path.exists()
    saved = json.loads(meta_path.read_text())
    assert saved["fingerprint"] == config_fingerprint(cfg)


def test_is_cache_fresh_detects_a_stale_cache_after_a_config_change(tmp_path, synthetic_nifti):
    path, _, _ = synthetic_nifti
    cfg_a = PreprocessConfig(target_size_hw=(224, 224))
    preprocess_one(path, out_dir=tmp_path, cfg=cfg_a, volume_id="synthetic_1_a_1")

    assert is_cache_fresh(tmp_path, "synthetic_1_a_1", cfg_a) is True

    cfg_b = PreprocessConfig(target_size_hw=(112, 112))  # a real setting change, e.g. different target size
    assert is_cache_fresh(tmp_path, "synthetic_1_a_1", cfg_b) is False  # must NOT be silently reused


def test_is_cache_fresh_false_when_nothing_cached(tmp_path):
    assert is_cache_fresh(tmp_path, "never_processed", PreprocessConfig()) is False


def test_the_compute_device_does_not_change_the_cache_fingerprint():
    # CPU and GPU resampling produce the same cache; switching must not look "stale"
    assert config_fingerprint(PreprocessConfig(device="cpu")) == config_fingerprint(PreprocessConfig(device="cuda"))


def test_the_sidecar_records_the_per_volume_stats_for_resuming(tmp_path, synthetic_nifti):
    path, _, _ = synthetic_nifti
    cfg = PreprocessConfig(target_size_hw=(64, 64))
    result = preprocess_one(path, out_dir=tmp_path, cfg=cfg, volume_id="v1")
    assert result.ok

    stats = load_cached_stats(tmp_path, "v1")
    assert stats["n_slices"] == result.n_slices
    assert stats["spacing_z_mm"] == result.spacing_after_resample[0]
    assert stats["crop_shape"] == "x".join(map(str, result.crop_shape))
    assert stats["npy_path"] == str(tmp_path / "v1.npy")
    assert load_cached_stats(tmp_path, "never_processed") is None


def test_is_cache_fresh_without_the_raw_file_trusts_the_fingerprint(tmp_path, synthetic_nifti):
    # after ingest the raw scan is deleted; the cache must still count as fresh
    path, _, _ = synthetic_nifti
    cfg = PreprocessConfig(target_size_hw=(64, 64))
    preprocess_one(path, out_dir=tmp_path, cfg=cfg, volume_id="v1")
    path.unlink()
    assert is_cache_fresh(tmp_path, "v1", cfg) is True
    assert is_cache_fresh(tmp_path, "v1", PreprocessConfig(target_size_hw=(32, 32))) is False
