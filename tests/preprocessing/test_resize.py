import numpy as np
import pytest

from ct_preprocessing.resize import resize_slices


@pytest.mark.parametrize("in_shape", [(5, 70, 55), (5, 300, 180), (5, 224, 224)])
def test_stretch_always_hits_exact_target_shape(in_shape):
    hu = np.random.default_rng(0).normal(size=in_shape).astype(np.float32)
    out = resize_slices(hu, target_hw=(224, 224), mode="stretch")
    assert out.shape == (in_shape[0], 224, 224)
    assert out.dtype == hu.dtype


def test_pad_preserves_aspect_ratio_and_hits_exact_target_shape():
    hu = np.zeros((5, 100, 50), dtype=np.float32)  # 2:1 aspect, tall
    hu[:, :, :] = -1000.0
    hu[:, 40:60, 15:35] = 40.0

    out = resize_slices(hu, target_hw=(224, 224), mode="pad")
    assert out.shape == (5, 224, 224)
    # borders should be background (the fill value), not stretched content
    assert np.isclose(out[0, 0, 0], -1000.0)


def test_unknown_mode_raises():
    hu = np.zeros((5, 20, 20), dtype=np.float32)
    with pytest.raises(ValueError):
        resize_slices(hu, target_hw=(224, 224), mode="nonsense")
