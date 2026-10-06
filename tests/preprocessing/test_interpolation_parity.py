"""The CPU (scipy) and GPU (torch) resampling paths must put their samples in the same places.

scipy's ``zoom`` reads output sample i at input position i*(n_in-1)/(n_out-1)
("corner aligned"). torch matches that only with ``align_corners=True``; with its
default (False) the arrays differ by up to about a quarter voxel, so a cache built
on one device would not equal inference on the other.

The torch comparisons run torch on the CPU (device="cpu"): they need torch installed
but no GPU, and are skipped without it.
"""
import numpy as np
import pytest
from scipy.ndimage import zoom


def corner_aligned_reference(arr, new_shape):
    """Linear interpolation with output sample i at input position i*(n_in-1)/(n_out-1), axis by axis."""
    out = arr.astype(np.float64)
    for axis, (n_in, n_out) in enumerate(zip(arr.shape, new_shape)):
        positions = np.linspace(0, n_in - 1, n_out)
        out = np.apply_along_axis(lambda v: np.interp(positions, np.arange(n_in), v), axis, out)
    return out


def test_scipy_zoom_is_corner_aligned():
    """The convention the GPU path has to match (no torch needed)."""
    arr = np.random.default_rng(0).normal(size=(7, 9, 11)).astype(np.float32)
    new_shape = (12, 14, 8)
    factors = np.array(new_shape) / np.array(arr.shape)
    np.testing.assert_allclose(zoom(arr, factors, order=1), corner_aligned_reference(arr, new_shape), atol=1e-4)


def test_the_gpu_resample_agrees_with_the_cpu_resample():
    pytest.importorskip("torch")
    from ct_preprocessing.spacing import _resample_gpu

    arr = np.random.default_rng(0).normal(size=(7, 9, 11)).astype(np.float32)
    new_shape = (12, 14, 8)
    cpu = zoom(arr, np.array(new_shape) / np.array(arr.shape), order=1)
    np.testing.assert_allclose(_resample_gpu(arr, new_shape, device="cpu"), cpu, atol=1e-4)

    ramp = np.tile(np.linspace(0, 400, 150, dtype=np.float32)[:, None, None], (1, 8, 8))  # a smooth HU ramp
    np.testing.assert_allclose(_resample_gpu(ramp, (300, 8, 8), device="cpu"), zoom(ramp, (2.0, 1, 1), order=1), atol=1e-3)


def test_the_gpu_resize_agrees_with_the_cpu_resize():
    pytest.importorskip("torch")
    from ct_preprocessing.resize import _resize_gpu

    arr = np.random.default_rng(0).normal(size=(5, 30, 40)).astype(np.float32)
    cpu = zoom(arr, (1.0, 16 / 30, 24 / 40), order=1)
    np.testing.assert_allclose(_resize_gpu(arr, 16, 24, device="cpu"), cpu, atol=1e-4)
