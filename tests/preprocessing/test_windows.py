import numpy as np

from ct_preprocessing.windows import DEFAULT_WINDOWS, apply_window, apply_windows


def test_apply_window_clips_and_scales_to_unit_range():
    hu = np.array([-2000.0, -1000.0, 0.0, 400.0, 2000.0], dtype=np.float32)
    out = apply_window(hu, lo=-1000.0, hi=400.0)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.isclose(out[0], 0.0)  # clipped below range
    assert np.isclose(out[1], 0.0)  # exactly at lo
    assert np.isclose(out[-1], 1.0)  # clipped above range
    assert np.isclose(out[-2], 1.0)  # exactly at hi


def test_apply_windows_stacks_three_channels_in_order():
    hu_slice = np.zeros((32, 32), dtype=np.float32)
    out = apply_windows(hu_slice, DEFAULT_WINDOWS)
    assert out.shape == (3, 32, 32)


def test_apply_windows_on_a_stack_of_slices():
    hu_stack = np.zeros((7, 32, 32), dtype=np.float32)
    out = apply_windows(hu_stack, DEFAULT_WINDOWS)
    assert out.shape == (7, 3, 32, 32)
    assert out.dtype == np.float32
