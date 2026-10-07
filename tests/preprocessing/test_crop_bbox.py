"""The crop box comes from per-axis reductions, not a coordinate for every voxel.

``np.nonzero(mask)`` builds a (z, y, x) coordinate for every foreground voxel -- about
50 bytes each, ~2.6 GB for one large chest CT -- to read off six numbers. The box
must be exactly the same, with a tiny fraction of the memory.
"""
import tracemalloc

import numpy as np

from ct_preprocessing.crop import foreground_bbox


def brute_force_box(hu, threshold_hu, margin):
    """The old definition: min/max of the coordinate of every foreground voxel."""
    coords = np.array(np.nonzero(hu > threshold_hu))
    mins, maxs = coords.min(axis=1), coords.max(axis=1) + 1
    return tuple(slice(max(0, int(lo) - margin), min(n, int(hi) + margin)) for lo, hi, n in zip(mins, maxs, hu.shape))


def test_the_box_is_identical_to_the_brute_force_box_on_random_volumes():
    rng = np.random.default_rng(1)
    for _ in range(40):
        hu = np.full((40, 60, 60), -1000.0, dtype=np.float32)
        z0, y0, x0 = rng.integers(0, 10, 3)
        hu[z0 : z0 + rng.integers(5, 25), y0 : y0 + rng.integers(10, 40), x0 : x0 + rng.integers(10, 40)] = 40.0
        hu[rng.integers(0, 40), rng.integers(0, 60), rng.integers(0, 60)] = 100.0  # a stray voxel
        margin = int(rng.integers(0, 6))
        got = foreground_bbox(hu, threshold_hu=-500.0, margin=margin, keep_largest_component=False)
        assert got == brute_force_box(hu, -500.0, margin)


def test_the_largest_component_path_still_ignores_a_stray_object():
    hu = np.full((40, 80, 80), -1000.0, dtype=np.float32)
    hu[5:35, 20:60, 20:60] = 40.0  # the body
    hu[0:2, 0:2, 0:2] = 40.0  # a far-away blob (a scanner-table fragment)
    box = foreground_bbox(hu, margin=0, cc_downsample=1)
    assert box == (slice(5, 35), slice(20, 60), slice(20, 60))


def test_a_single_voxel_volume_gives_a_one_voxel_box():
    hu = np.full((8, 8, 8), -1000.0, dtype=np.float32)
    hu[3, 4, 5] = 40.0
    assert foreground_bbox(hu, margin=0, keep_largest_component=False) == (slice(3, 4), slice(4, 5), slice(5, 6))


def test_the_box_needs_far_less_memory_than_a_coordinate_array():
    hu = np.full((120, 200, 200), -1000.0, dtype=np.float32)
    hu[:, 20:180, 20:180] = 40.0  # 3 million foreground voxels: ~150 MB as (z, y, x) coordinates
    tracemalloc.start()
    foreground_bbox(hu, keep_largest_component=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 20e6  # just the boolean mask (4.8 MB) and a few short vectors
