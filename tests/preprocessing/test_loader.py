import numpy as np
import pytest

from ct_preprocessing.loader import check_hu_plausible, ensure_calibrated_hu, load_nifti_volume


def test_load_nifti_volume_roundtrips_shape_and_spacing(synthetic_nifti):
    path, expected_hu, expected_spacing = synthetic_nifti

    volume = load_nifti_volume(path, volume_id="synthetic_1_a_1")

    assert volume.hu.shape == expected_hu.shape
    assert volume.spacing == pytest.approx(expected_spacing, abs=1e-4)
    assert volume.orientation == "RAS+"
    # body / lung / background values should all survive the round trip
    assert np.isclose(volume.hu.max(), 40.0, atol=1.0)
    assert np.isclose(volume.hu.min(), -1000.0, atol=1.0)


def test_check_hu_plausible_flags_real_hu(synthetic_hu):
    report = check_hu_plausible(synthetic_hu)
    assert report["looks_like_hu"] is True
    assert report["min"] < -500


def test_check_hu_plausible_flags_unconverted_values():
    # Simulate a file where slope/intercept was never applied: values sit
    # near 0-4000 instead of -1000-ish for air.
    fake_raw = np.full((10, 20, 20), 1200.0, dtype=np.float32)
    report = check_hu_plausible(fake_raw)
    assert report["looks_like_hu"] is False


def test_ensure_calibrated_hu_leaves_real_hu_alone(synthetic_hu):
    out, info = ensure_calibrated_hu(synthetic_hu, rescale_slope=None, rescale_intercept=None)
    assert info["already_calibrated"] is True
    assert info["applied"] is False
    np.testing.assert_array_equal(out, synthetic_hu)


def test_ensure_calibrated_hu_leaves_scanner_specific_low_min_alone():
    # Confirmed real CT-RATE case: air legitimately at -8192 for some
    # scanners -- must not be "corrected" further.
    hu = np.full((10, 20, 20), -8192.0, dtype=np.float32)
    hu[:, 5:15, 5:15] = -100.0
    out, info = ensure_calibrated_hu(hu, rescale_slope=1.0, rescale_intercept=-8192.0)
    assert info["already_calibrated"] is True
    np.testing.assert_array_equal(out, hu)


def test_ensure_calibrated_hu_applies_rescale_when_uncalibrated():
    # Raw, un-rescaled 12-bit detector output: air ~0, tissue ~1024ish,
    # needing RescaleIntercept=-1024 to become real HU.
    raw = np.full((10, 20, 20), 0.0, dtype=np.float32)
    raw[:, 5:15, 5:15] = 1064.0  # -> 40 HU after -1024
    out, info = ensure_calibrated_hu(raw, rescale_slope=1.0, rescale_intercept=-1024.0)
    assert info["applied"] is True
    assert np.isclose(out.min(), -1024.0)
    assert np.isclose(out.max(), 40.0)


def test_ensure_calibrated_hu_reports_when_uncalibrated_and_no_rescale_given():
    raw = np.full((10, 20, 20), 0.0, dtype=np.float32)
    out, info = ensure_calibrated_hu(raw, rescale_slope=None, rescale_intercept=None)
    assert info["applied"] is False
    assert info["already_calibrated"] is False
    np.testing.assert_array_equal(out, raw)  # unchanged -- left for QC to flag
