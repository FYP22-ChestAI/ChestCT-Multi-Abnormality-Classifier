"""Unusable values, the HU floor and the calibration record.

The cache stores int16. Casting NaN/Inf gives a silent 0 and casting 40000 wraps
to -25536, and QC used to pass both -- so those scans must fail BEFORE the cast.
Scanner padding (-8192 on Siemens go.All) is floored to -1024, and how the HU
were obtained is written to the sidecar because the raw file is deleted.
"""
import json

import nibabel as nib
import numpy as np

from ct_preprocessing.pipeline import process_scan
from ct_preprocessing.preprocess import config_fingerprint, preprocess_one
from ct_preprocessing.preprocess_config import PreprocessConfig
from ct_preprocessing.quality import QCThresholds, check_volume, int16_problem, nonfinite_problem


def phantom(shape=(24, 48, 48)):
    """Air background, a tissue block and a lung-like region, in true HU."""
    a = np.full(shape, -1000.0, dtype=np.float32)
    a[4:20, 8:40, 8:40] = 40.0
    a[8:16, 16:32, 16:32] = -800.0
    return a


def padded_phantom():
    a = phantom()
    a[4:20, 8:14, 8:14] = -8192.0  # padding inside the body's box, as outside the field of view on go.All
    return a


def save(tmp_path, array, name="v.nii.gz"):
    path = tmp_path / name
    nib.save(nib.Nifti1Image(np.transpose(array, (2, 1, 0)).astype(np.float32), np.eye(4)), str(path))
    return path


def run(tmp_path, array, name="v.nii.gz", rescale=None, **cfg_kwargs):
    cfg = PreprocessConfig(target_spacing_zyx=(1.0, 1.0, 1.0), target_size_hw=(32, 32), crop_margin=2, **cfg_kwargs)
    slope, intercept = rescale or (None, None)
    return process_scan(
        save(tmp_path, array, name), cfg, volume_id="v", rescale_slope=slope, rescale_intercept=intercept,
        qc_thresholds=QCThresholds(min_slices=5), scan_format="nifti",
    )


# ------------------------------------------------ unusable values fail before the int16 cast
def test_a_clean_phantom_passes(tmp_path):
    result = run(tmp_path, phantom())
    assert result.error is None and result.qc.passed


def test_a_nan_voxel_fails_the_scan_instead_of_becoming_zero(tmp_path):
    a = phantom()
    a[10, 24, 24] = np.nan
    result = run(tmp_path, a)
    assert result.hu is None and result.qc is None
    assert "NaN or infinite" in result.error


def test_an_infinite_voxel_fails_the_scan(tmp_path):
    a = phantom()
    a[10, 24, 24] = np.inf
    result = run(tmp_path, a)
    assert result.hu is None and "NaN or infinite" in result.error


def test_values_beyond_int16_fail_the_scan_instead_of_wrapping(tmp_path):
    a = phantom()
    a[6:18, 12:36, 12:36] = 40000.0  # would wrap to -25536 in an int16 cast
    result = run(tmp_path, a)
    assert result.hu is None
    assert "do not fit in int16" in result.error


def test_real_metal_range_values_are_kept(tmp_path):
    a = phantom()
    a[6:18, 12:36, 12:36] = 3000.0  # a large implant: legitimate, must be neither clipped nor rejected
    result = run(tmp_path, a)
    assert result.error is None and result.qc.passed
    assert result.hu.max() >= 2900


def test_interpolated_values_are_rounded_not_truncated(tmp_path):
    a = phantom()
    a[4:20, 8:40, 8:40] = 40.6  # truncation would store 40
    result = run(tmp_path, a)
    assert result.error is None
    assert (result.hu == 41).any() and not (result.hu == 40).any()


def test_the_helpers():
    ok = np.array([-8192.0, 0.0, 3071.0, 32767.0])
    assert nonfinite_problem(ok) is None and int16_problem(ok) is None
    assert "NaN or infinite" in nonfinite_problem(np.array([1.0, np.nan]))
    assert "NaN or infinite" in int16_problem(np.array([1.0, -np.inf]))
    assert "do not fit in int16" in int16_problem(np.array([0.0, 40000.0]))
    assert "do not fit in int16" in int16_problem(np.array([-40000.0, 0.0]))
    assert nonfinite_problem(np.zeros((0, 4, 4))) == "empty volume"


# -------------------------------------------------- check_volume no longer assumes tidy input
def good(n=100):
    a = np.full((n, 16, 16), -1000.0, dtype=np.float32)
    a[:, 4:12, 4:12] = 40.0
    return a


def test_float_input_with_infinity_fails_qc():
    a = good()
    a[5, 5, 5] = np.inf
    result = check_volume("v", a, (1.5, 0.75, 0.75))
    assert not result.passed and any("NaN or infinite" in r for r in result.reasons)


def test_float_input_with_nan_fails_qc():
    a = good()
    a[5, 5, 5] = np.nan
    assert not check_volume("v", a, (1.5, 0.75, 0.75)).passed


def test_a_wrong_rank_or_empty_array_fails_qc_with_a_reason_instead_of_raising():
    for bad in (np.zeros((10, 10)), np.zeros((0, 16, 16)), np.zeros(5)):
        result = check_volume("v", bad, (1.5, 0.75, 0.75))
        assert not result.passed and "non-empty (N, H, W)" in result.reasons[0]


def test_infinite_spacing_is_invalid():
    result = check_volume("v", good(), (float("inf"), 0.75, 0.75))
    assert not result.passed and any("invalid spacing" in r for r in result.reasons)


# ------------------------------------------------------------- the HU floor (scanner padding)
def test_padding_is_floored_to_minus_1024_by_default(tmp_path):
    result = run(tmp_path, padded_phantom())
    assert result.error is None
    assert result.hu.min() == -1024
    assert result.qc.passed


def test_the_floor_leaves_lung_and_tissue_alone(tmp_path):
    padded = run(tmp_path, padded_phantom(), name="p.nii.gz")
    clean = run(tmp_path, phantom(), name="c.nii.gz")
    for value in (-800, 40):
        assert (padded.hu == value).any() and (clean.hu == value).any()


def test_the_floor_can_be_switched_off(tmp_path):
    result = run(tmp_path, padded_phantom(), hu_floor=None)
    assert result.hu.min() < -4000  # still deep padding: nothing was floored


def test_the_floor_is_part_of_the_cache_fingerprint():
    assert config_fingerprint(PreprocessConfig(hu_floor=-1024)) == config_fingerprint(PreprocessConfig(hu_floor=-1024.0))
    assert config_fingerprint(PreprocessConfig(hu_floor=-1024)) != config_fingerprint(PreprocessConfig(hu_floor=None))


# ------------------------------------------------------------------ the calibration record
def test_the_calibration_decision_is_recorded_for_an_already_hu_scan(tmp_path):
    calib = run(tmp_path, padded_phantom()).calibration
    assert calib["already_calibrated"] is True and calib["applied"] is False
    assert calib["hu_range_before_floor"] == [-8192.0, 40.0]
    assert calib["hu_floor"] == -1024.0 and calib["floored"] is True


def test_the_calibration_decision_is_recorded_when_a_rescale_is_applied(tmp_path):
    raw = phantom() + 1024.0  # raw stored values: air 24, tissue 1064
    result = run(tmp_path, raw, rescale=(1.0, -1024.0))
    assert result.calibration["applied"] is True and result.calibration["rescale_intercept"] == -1024.0
    assert result.hu.min() == -1000 and result.qc.passed


def test_the_sidecar_carries_the_calibration_record(tmp_path):
    out_dir = tmp_path / "cache"
    cfg = PreprocessConfig(target_spacing_zyx=(1.0, 1.0, 1.0), target_size_hw=(32, 32), crop_margin=2)
    result = preprocess_one(save(tmp_path, padded_phantom()), out_dir=out_dir, cfg=cfg, volume_id="v1")
    assert result.ok, result.error

    sidecar = json.loads((out_dir / "v1.meta.json").read_text())
    assert sidecar["version"] == "m1-v3"
    assert sidecar["calibration"]["already_calibrated"] is True
    assert sidecar["calibration"]["hu_range_before_floor"][0] == -8192.0
    assert sidecar["calibration"]["floored"] is True


def test_a_scan_with_unusable_values_is_a_failed_result_and_leaves_no_cache(tmp_path):
    a = phantom()
    a[10, 24, 24] = np.nan
    out_dir = tmp_path / "cache"
    result = preprocess_one(
        save(tmp_path, a), out_dir=out_dir, cfg=PreprocessConfig(target_size_hw=(32, 32)), volume_id="nan_1"
    )
    assert not result.ok and "NaN" in result.error
    assert not (out_dir / "nan_1.npy").exists() and not (out_dir / "nan_1.meta.json").exists()
