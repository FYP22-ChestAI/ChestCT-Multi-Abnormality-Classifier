"""run_inference takes its settings from the YAML the cache was built with, and refuses a mismatch.

Every test writes its own config and its own fingerprint record under tmp_path, so none
depends on whatever data/ happens to exist on the machine running the tests.
"""
import json

import pytest

from ct_preprocessing.config import load_config
from ct_preprocessing.inference import ConfigMismatch, check_matches_cache, run_inference
from ct_preprocessing.preprocess import config_fingerprint
from ct_preprocessing.preprocess_config import PreprocessConfig
from ct_preprocessing.quality import QCThresholds


def write_config(tmp_path, size=64, record_fingerprint=None):
    record = tmp_path / "preprocessing_manifest.json"
    if record_fingerprint is not None:
        record.write_text(json.dumps({"fingerprint": record_fingerprint}))
    config = tmp_path / "preprocessing.yaml"
    config.write_text(
        "\n".join([
            "paths:",
            f"  preprocessing_manifest_path: {record.as_posix()}",
            "preprocess:",
            f"  target_size_hw: [{size}, {size}]",
            "  hu_floor: -1024",
            "qc:",
            "  min_slices: 10",
        ])
        + "\n"
    )
    return config


def test_with_no_cfg_the_yaml_settings_and_qc_thresholds_are_used(synthetic_nifti, tmp_path):
    path, _, _ = synthetic_nifti
    [out] = run_inference(path, config_path=write_config(tmp_path, size=48))
    assert out.passed  # min_slices 10 came from the YAML: the default 80 would fail the 40-slice fixture
    assert out.hu.shape == (40, 48, 48)  # target_size_hw 48 came from the YAML too


def test_a_yaml_that_no_longer_matches_the_cache_is_refused(synthetic_nifti, tmp_path):
    path, _, _ = synthetic_nifti
    config = write_config(tmp_path, size=48, record_fingerprint="0123456789abcdef")
    with pytest.raises(ConfigMismatch, match="differ from the ones the cache"):
        run_inference(path, config_path=config)
    [out] = run_inference(path, config_path=config, check_cache=False)  # an explicit opt-out
    assert out.passed


def test_a_yaml_that_matches_the_recorded_fingerprint_is_accepted(synthetic_nifti, tmp_path):
    path, _, _ = synthetic_nifti
    fingerprint = config_fingerprint(load_config(write_config(tmp_path, size=48)).preprocess)
    [out] = run_inference(path, config_path=write_config(tmp_path, size=48, record_fingerprint=fingerprint))
    assert out.passed


def test_without_a_cache_record_there_is_nothing_to_compare(synthetic_nifti, tmp_path):
    path, _, _ = synthetic_nifti
    [out] = run_inference(path, config_path=write_config(tmp_path, size=48))  # no record file written
    assert out.passed


def test_the_device_does_not_count_as_a_difference(tmp_path):
    bare = PreprocessConfig(target_size_hw=(48, 48))
    data_cfg = load_config(write_config(tmp_path, size=48, record_fingerprint=config_fingerprint(bare)))
    check_matches_cache(PreprocessConfig(target_size_hw=(48, 48), device="cuda"), data_cfg)  # no raise


def test_an_explicit_cfg_is_used_as_given_and_not_checked(synthetic_nifti, tmp_path):
    path, _, _ = synthetic_nifti
    write_config(tmp_path, record_fingerprint="0123456789abcdef")  # a record that would never match
    [out] = run_inference(path, cfg=PreprocessConfig(target_size_hw=(40, 40)), qc_thresholds=QCThresholds(min_slices=10))
    assert out.hu.shape == (40, 40, 40)


def test_a_missing_config_says_how_to_fix_it(synthetic_nifti, tmp_path):
    path, _, _ = synthetic_nifti
    with pytest.raises(FileNotFoundError, match="config_path= or cfg="):
        run_inference(path, config_path=tmp_path / "nope.yaml")
