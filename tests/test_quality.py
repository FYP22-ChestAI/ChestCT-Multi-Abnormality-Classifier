import numpy as np

from chestct.data.quality import QCThresholds, check_volume, save_montage


def _good_array(n=200):
    rng = np.random.default_rng(0)
    # clipped to a physically plausible HU range (real air floor is -1024;
    # dense bone/contrast tops out well under our 3200 "implausible" ceiling)
    arr = rng.normal(loc=-400, scale=300, size=(n, 32, 32))
    arr = np.clip(arr, -1024, 1000).astype(np.int16)
    arr[:, :4, :4] = -1024  # some real background air present
    return arr


def test_check_volume_passes_a_clean_scan():
    result = check_volume("v1", _good_array(), spacing=(1.5, 0.75, 0.75))
    assert result.passed
    assert result.reasons == []


def test_check_volume_flags_too_few_slices():
    result = check_volume("v1", _good_array(n=5), spacing=(1.5, 0.75, 0.75))
    assert not result.passed
    assert any("slice count" in r for r in result.reasons)


def test_check_volume_flags_nan():
    arr = _good_array().astype(np.float32)
    arr[0, 0, 0] = np.nan
    result = check_volume("v1", arr, spacing=(1.5, 0.75, 0.75))
    assert not result.passed
    assert any("NaN" in r for r in result.reasons)


def test_check_volume_does_not_flag_a_high_max():
    # A high max (metal, contrast, dense implants) is a normal clinical finding,
    # not a defect -- confirmed against real CT-RATE data (12.3% of volumes are
    # positive for "Medical material") and real chest CT preprocessing practice,
    # which clips it via windowing rather than treating it as an anomaly. No
    # check for it at all -- see docs/data_contract.md.
    arr = _good_array()
    arr[0, 0, 0] = 9000
    result = check_volume("v1", arr, spacing=(1.5, 0.75, 0.75))
    assert result.passed
    assert result.flags == []


def test_check_volume_flags_uncalibrated_min():
    # The real defect signature: a minimum suspiciously near/above 0, i.e.
    # raw un-rescaled detector values, not real HU.
    arr = _good_array()
    arr = arr - arr.min()  # shift so min is 0 -- mimics uncalibrated raw data
    result = check_volume("v1", arr, spacing=(1.5, 0.75, 0.75))
    assert not result.passed
    assert any("looks uncalibrated" in r for r in result.reasons)


def test_check_volume_passes_scanner_specific_low_min():
    # Confirmed real CT-RATE data: some scanners legitimately use a very
    # different (but correctly-applied) RescaleIntercept, e.g. -8192 instead
    # of -1024 -- must NOT be flagged as implausible.
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=-7200, scale=300, size=(200, 32, 32))
    arr = np.clip(arr, -8192, -6000).astype(np.int16)
    arr[:, :4, :4] = -8192
    result = check_volume("v1", arr, spacing=(1.5, 0.75, 0.75))
    assert result.passed


def test_check_volume_flags_constant_volume():
    arr = np.full((200, 32, 32), -500, dtype=np.int16)
    result = check_volume("v1", arr, spacing=(1.5, 0.75, 0.75))
    assert not result.passed
    assert any("near-constant" in r for r in result.reasons)


def test_check_volume_flags_missing_spacing():
    result = check_volume("v1", _good_array(), spacing=(None, 0.75, 0.75))
    assert not result.passed
    assert any("spacing" in r for r in result.reasons)


def test_check_volume_flags_unusual_spacing_without_excluding():
    result = check_volume("v1", _good_array(), spacing=(5.0, 0.75, 0.75))
    assert result.passed  # flagged, not excluded
    assert any("unusual z-spacing" in f for f in result.flags)


def test_save_montage_writes_a_file(tmp_path):
    out_path = tmp_path / "montage.png"
    save_montage(_good_array(), out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0
