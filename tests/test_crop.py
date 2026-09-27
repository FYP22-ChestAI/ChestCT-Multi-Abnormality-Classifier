"""Directly tests the "what about air in the middle?" question: cropping
must keep the whole body even though the lungs read as background-like air.
"""
import numpy as np

from chestct.data.crop import crop_to_foreground, foreground_bbox
from chestct.data.types import Volume


def test_bbox_covers_full_body_despite_internal_air(synthetic_hu):
    box = foreground_bbox(synthetic_hu, threshold_hu=-500.0, margin=0)

    # every z-slice has body tissue -> the z box should span (almost) the
    # whole volume, not just the lung-free top/bottom slices (there are none
    # in this fixture, but the box must not have collapsed to a thin sliver)
    z_slice = box[0]
    assert (z_slice.stop - z_slice.start) >= synthetic_hu.shape[0] - 1

    # the box must be much larger than either lung individually, i.e. it's
    # bounded by the outer body contour, not by the lung shape
    y_slice, x_slice = box[1], box[2]
    body_width = x_slice.stop - x_slice.start
    assert body_width > synthetic_hu.shape[2] * 0.5


def test_crop_does_not_alter_lung_hu_values(synthetic_hu):
    volume = Volume(hu=synthetic_hu, spacing=(1.5, 1.0, 1.0), orientation="RAS+")
    cropped = crop_to_foreground(volume, threshold_hu=-500.0, margin=4)

    # the low-HU lung regions must still be present, at their true value,
    # inside the cropped output -- cropping must never zero/mask them out
    assert np.any(np.isclose(cropped.hu, -800.0, atol=1.0))
    # and the background OUTSIDE the body should be gone (or nearly so)
    assert np.mean(np.isclose(cropped.hu, -1000.0, atol=1.0)) < np.mean(
        np.isclose(synthetic_hu, -1000.0, atol=1.0)
    )


def test_crop_returns_full_volume_when_nothing_above_threshold():
    all_air = np.full((10, 20, 20), -1000.0, dtype=np.float32)
    box = foreground_bbox(all_air, threshold_hu=-500.0)
    assert all(s.start == 0 and s.stop == dim for s, dim in zip(box, all_air.shape))


def test_bbox_ignores_stray_artifact_via_downsampled_cc():
    # A small, disconnected bright blob far from the body (e.g. part of the
    # scanner table) must not stretch the box -- verified here specifically
    # through the downsampled connected-component path (cc_downsample>1),
    # which is the fix for the OOM a full-resolution label() pass caused on
    # a large real volume.
    hu = np.full((20, 80, 80), -1000.0, dtype=np.float32)
    hu[5:15, 30:50, 30:50] = 40.0  # the real body, centred
    hu[0:2, 0:4, 0:4] = 40.0  # a small stray artifact in a far corner

    box = foreground_bbox(hu, threshold_hu=-500.0, margin=2, cc_downsample=4)

    # the box should track the body, not stretch out to the corner artifact
    assert box[1].start > 10  # well inside the body's actual y-range, not near 0
    assert box[2].start > 10


def test_crop_margin_is_respected_and_clamped():
    hu = np.full((10, 20, 20), -1000.0, dtype=np.float32)
    hu[3:7, 5:15, 5:15] = 40.0  # a small tissue block away from the edges

    box = foreground_bbox(hu, threshold_hu=-500.0, margin=100)  # huge margin
    # margin must be clamped to the array bounds, not go negative/out of range
    assert all(s.start == 0 for s in box)
    assert box[0].stop == 10 and box[1].stop == 20 and box[2].stop == 20
