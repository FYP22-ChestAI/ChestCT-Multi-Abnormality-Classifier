"""Step 6: HU windowing -- clip and rescale HU to [0, 1] picture intensity.

This module only defines and tests the windowing function. It does NOT run
during preprocessing: the cached .npy holds raw, unwindowed HU (see
preprocess.py). Windowing runs later, every time the Dataset (dataset.py)
loads slices -- so the windows can still be changed during development
without re-processing the whole cache, and get fixed only before final
evaluation (per the proposal).
"""
from __future__ import annotations

import numpy as np

# Three windows used as the 3 input channels, following the AnyMC3D / CT-RATE
# convention: lung, soft tissue, all tissue. Also matches the 3-channel input
# a standard (ImageNet-pretrained) 2D encoder expects.
DEFAULT_WINDOWS: dict[str, tuple[float, float]] = {
    "lung": (-1000.0, 400.0),
    "soft_tissue": (-150.0, 250.0),
    "all_tissue": (-1000.0, 1000.0),
}


def apply_window(hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip HU to [lo, hi] and rescale linearly to [0, 1] float32."""
    clipped = np.clip(hu, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def apply_windows(hu_slices: np.ndarray, windows: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
    """Turn HU slices into stacked window channels, each scaled to [0, 1].

    ``hu_slices`` may be a single slice (H, W) -> returns (3, H, W), or a
    stack (K, H, W) -> returns (K, 3, H, W). The channel axis is always
    inserted right after any leading slice axis.
    """
    windows = windows or DEFAULT_WINDOWS
    channels = [apply_window(hu_slices, lo, hi) for lo, hi in windows.values()]
    return np.stack(channels, axis=-3)
