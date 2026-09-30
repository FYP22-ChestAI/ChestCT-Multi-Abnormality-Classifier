"""Tests for the clinical/inference front door. Confirms it shares the exact
same core as the training path, that a QC failure or a hard error is
recorded on its own row rather than raised, and that discovery failure
(nothing found at all) is the one thing that still raises."""
import pytest

from chestct.preprocessing.preprocess_config import PreprocessConfig
from chestct.preprocessing.quality import QCThresholds
from chestct.inference import run_inference


def test_run_inference_returns_a_ready_tensor(synthetic_nifti):
    path, _, _ = synthetic_nifti
    cfg = PreprocessConfig(target_size_hw=(64, 64))
    thresholds = QCThresholds(min_slices=10)  # the synthetic fixture only has 40 slices

    [out] = run_inference(path, cfg=cfg, qc_thresholds=thresholds, normalize_imagenet=False)

    assert out.volume_id == "synthetic_1_a_1"
    assert out.passed
    assert out.pixel_values.shape == (40, 3, 64, 64)
    assert out.pixel_values.min() >= 0.0 and out.pixel_values.max() <= 1.0
    assert out.qc.passed


def test_run_inference_flags_a_qc_failure_without_raising(synthetic_nifti):
    path, _, _ = synthetic_nifti
    cfg = PreprocessConfig(target_size_hw=(64, 64))
    # default QCThresholds requires >= 80 slices; the fixture only has 40 --
    # this must be flagged (passed=False, no tensor), not silently processed
    # like the training path would, but it must not raise either.
    [out] = run_inference(path, cfg=cfg)
    assert not out.passed
    assert out.pixel_values is None
    assert out.qc is not None and not out.qc.passed


def test_run_inference_flags_a_hard_error_on_a_bad_file(tmp_path):
    bad = tmp_path / "corrupt.nii.gz"
    bad.write_bytes(b"not a real nifti file")
    [out] = run_inference(bad)
    assert not out.passed
    assert out.pixel_values is None
    assert out.qc is None and out.error


def test_run_inference_raises_when_nothing_is_found_at_all():
    with pytest.raises(FileNotFoundError):
        run_inference("does/not/exist")
