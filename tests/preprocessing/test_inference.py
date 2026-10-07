"""Tests for the clinical/inference front door. Confirms it shares the exact
same core as the training path, that a QC failure or a hard error is
recorded on its own row rather than raised, and that discovery failure
(nothing found at all) is the one thing that still raises."""
import numpy as np
import pytest

from ct_preprocessing.preprocess_config import PreprocessConfig
from ct_preprocessing.quality import QCThresholds
from ct_preprocessing.inference import run_inference


def test_run_inference_returns_the_cached_form_of_the_volume(synthetic_nifti):
    path, _, _ = synthetic_nifti
    cfg = PreprocessConfig(target_size_hw=(64, 64))
    thresholds = QCThresholds(min_slices=10)  # the synthetic fixture only has 40 slices

    [out] = run_inference(path, cfg=cfg, qc_thresholds=thresholds)

    assert out.volume_id == "synthetic_1_a_1"
    assert out.passed
    assert out.hu.shape == (40, 64, 64) and out.hu.dtype == np.int16  # raw HU, unwindowed, as cached
    assert out.hu.min() < -500 < 0 < out.hu.max()  # air and tissue both present: not scaled to [0, 1]
    assert out.spacing_zyx is not None
    assert out.qc.passed


def test_run_inference_flags_a_qc_failure_without_raising(synthetic_nifti):
    path, _, _ = synthetic_nifti
    cfg = PreprocessConfig(target_size_hw=(64, 64))
    # default QCThresholds requires >= 80 slices; the fixture only has 40 --
    # this must be flagged (passed=False, no volume), not silently processed
    # like the training path would, but it must not raise either.
    [out] = run_inference(path, cfg=cfg)
    assert not out.passed
    assert out.hu is None
    assert out.qc is not None and not out.qc.passed


def test_run_inference_flags_a_hard_error_on_a_bad_file(tmp_path):
    bad = tmp_path / "corrupt.nii.gz"
    bad.write_bytes(b"not a real nifti file")
    [out] = run_inference(bad, cfg=PreprocessConfig())
    assert not out.passed
    assert out.hu is None
    assert out.qc is None and out.error


def test_run_inference_raises_when_nothing_is_found_at_all():
    with pytest.raises(FileNotFoundError):
        run_inference("does/not/exist", cfg=PreprocessConfig())
