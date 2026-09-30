"""Step 8: automatic quality checks + a montage picture, run per cached scan.

Each automatic check exists to catch one specific, real failure mode, on one
scan, before it silently reaches training. Automatic checks cannot catch a
scan that is numerically fine but visually wrong (upside-down, wrong body
part, motion blur) -- that's what the montage image is for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QCThresholds:
    min_slices: int = 80
    max_slices: int = 650
    # A minimum ABOVE this is the real defect signature: raw, uncalibrated
    # detector values (the 0-4095 range) never dip below roughly -500. Real
    # calibrated air legitimately varies by scanner (-1024 to -8192 or lower
    # have both been confirmed on real CT-RATE data) -- so this check no
    # longer assumes -1024 is universal (see docs/data_contract.md).
    hu_min_uncalibrated: float = -500.0
    min_std: float = 1.0  # "not all-black / not constant"
    z_spacing_low: float = 0.3
    z_spacing_high: float = 3.0


@dataclass
class QCResult:
    volume_id: str
    passed: bool
    reasons: list[str] = field(default_factory=list)  # why excluded; empty if passed
    flags: list[str] = field(default_factory=list)  # outliers noted but NOT excluded
    stats: dict = field(default_factory=dict)


def check_volume(
    volume_id: str,
    npy_array: np.ndarray,
    spacing: tuple | None,
    thresholds: QCThresholds | None = None,
) -> QCResult:
    t = thresholds or QCThresholds()
    reasons: list[str] = []
    flags: list[str] = []

    n = npy_array.shape[0]
    if n < t.min_slices or n > t.max_slices:
        reasons.append(f"slice count {n} outside [{t.min_slices}, {t.max_slices}]")

    has_nan = bool(np.isnan(npy_array).any()) if np.issubdtype(npy_array.dtype, np.floating) else False
    if has_nan:
        reasons.append("contains NaN")

    finite = npy_array[np.isfinite(npy_array)] if has_nan else npy_array
    if finite.size == 0:
        reasons.append("no finite values")
        vmin = vmax = vstd = float("nan")
    else:
        vmin, vmax, vstd = float(finite.min()), float(finite.max()), float(finite.std())
        if vmin > t.hu_min_uncalibrated:
            reasons.append(
                f"HU minimum too high ({vmin:.0f}) -- looks uncalibrated (raw values, not true HU)"
            )
        if vstd < t.min_std:
            reasons.append(f"near-constant volume (std={vstd:.3f})")

    if spacing is None or any(s is None or (isinstance(s, float) and np.isnan(s)) or s <= 0 for s in spacing):
        reasons.append(f"missing/invalid spacing: {spacing}")
    elif spacing[0] > t.z_spacing_high or spacing[0] < t.z_spacing_low:
        flags.append(f"unusual z-spacing: {spacing[0]:.2f} mm")

    return QCResult(
        volume_id=volume_id,
        passed=len(reasons) == 0,
        reasons=reasons,
        flags=flags,
        stats={"n_slices": n, "hu_min": vmin, "hu_max": vmax, "hu_std": vstd},
    )


def save_montage(npy_array: np.ndarray, out_path, n_tiles: int = 9, window: tuple[float, float] = (-1000, 400)) -> None:
    """Save an evenly-spaced grid of slices as a PNG, windowed for visibility.

    Exists because the numeric checks above cannot catch a scan that is
    valid numbers but visually wrong.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = npy_array.shape[0]
    idx = np.linspace(0, n - 1, num=min(n_tiles, n)).astype(int)
    cols = int(np.ceil(np.sqrt(len(idx))))
    rows = int(np.ceil(len(idx) / cols))

    lo, hi = window
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, idx):
        img = np.clip(npy_array[i], lo, hi)
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        ax.set_title(str(int(i)), fontsize=8)
        ax.axis("off")
    for ax in axes[len(idx) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
