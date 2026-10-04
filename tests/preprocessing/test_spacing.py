import numpy as np
import pytest

from ct_preprocessing.spacing import resample_to_spacing
from ct_preprocessing.types import Volume


def test_resample_upsamples_bigger_bodies_more_than_smaller_ones():
    """Worked example from docs/preprocessing/data_contract.md: a bigger physical body
    should end up with MORE voxels after resampling to a common spacing, not
    fewer -- spacing standardises mm/voxel, not the final voxel count."""
    target = (0.75, 0.75, 0.75)

    small = Volume(hu=np.zeros((10, 224, 224), dtype=np.float32), spacing=(1.5, 1.0, 1.0))
    big = Volume(hu=np.zeros((10, 252, 252), dtype=np.float32), spacing=(1.5, 1.5, 1.5))

    small_out = resample_to_spacing(small, target, min_change_ratio=0.0)
    big_out = resample_to_spacing(big, target, min_change_ratio=0.0)

    assert big_out.hu.shape[1] > small_out.hu.shape[1]
    for s in small_out.spacing:
        assert s == pytest.approx(0.75, abs=0.05)
    for s in big_out.spacing:
        assert s == pytest.approx(0.75, abs=0.05)


def test_resample_skips_axes_already_close_to_target():
    v = Volume(hu=np.zeros((10, 20, 20), dtype=np.float32), spacing=(0.75, 0.75, 0.75))
    out = resample_to_spacing(v, target_spacing_zyx=(0.75, 0.75, 0.75), min_change_ratio=0.05)
    assert out.hu.shape == v.hu.shape
    assert out.spacing == v.spacing
