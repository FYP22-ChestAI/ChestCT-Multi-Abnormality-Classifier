"""Tests for the shared core (pipeline.process_scan) that both the training
batch path and the future single-scan inference entry point call."""
import numpy as np

from chestct.data.pipeline import process_scan
from chestct.data.preprocess_config import PreprocessConfig
from chestct.data.quality import QCThresholds


def test_process_scan_end_to_end(synthetic_nifti):
    path, _, _ = synthetic_nifti
    cfg = PreprocessConfig(target_size_hw=(224, 224))
    # the synthetic fixture only has 40 slices; relax the default min_slices=80
    # so this test is about whether the pipeline runs correctly, not slice count
    thresholds = QCThresholds(min_slices=10)

    result = process_scan(path, cfg, volume_id="synthetic_1_a_1", qc_thresholds=thresholds)

    assert result.error is None
    assert result.hu is not None
    assert result.hu.shape[1:] == (224, 224)
    assert result.hu.dtype == np.int16
    assert result.qc is not None
    assert result.qc.passed  # synthetic fixture is already real HU, well-formed


def test_process_scan_always_returns_hu_even_when_qc_fails(tmp_path):
    # An uncalibrated (raw-looking) synthetic volume: QC should fail, but
    # the array must still be returned -- the caller decides what to do
    # with a QC failure (see docs/data_contract.md), process_scan never
    # withholds data on a soft QC failure, only on a hard error.
    import nibabel as nib

    raw = np.full((10, 20, 20), 0.0, dtype=np.float32)
    raw[:, 5:15, 5:15] = 1064.0
    path = tmp_path / "uncalibrated.nii.gz"
    nib.save(nib.Nifti1Image(np.transpose(raw, (2, 1, 0)), np.eye(4)), str(path))

    cfg = PreprocessConfig(target_size_hw=(32, 32))
    result = process_scan(path, cfg, volume_id="v1")  # no rescale given -> stays uncalibrated

    assert result.error is None
    assert result.hu is not None
    assert result.qc is not None
    assert not result.qc.passed


def test_process_scan_records_hard_errors():
    result = process_scan("does/not/exist.nii.gz", PreprocessConfig(), volume_id="missing")
    assert result.error is not None
    assert result.hu is None
    assert result.qc is None
