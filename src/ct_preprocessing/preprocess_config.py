"""``PreprocessConfig`` lives in its own tiny module so both ``preprocess.py``
(training-side, persists to disk) and ``pipeline.py`` (the shared core used
by training AND the future inference entry point) can import it without a
circular import between the two.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    target_spacing_zyx: tuple[float, float, float] = (1.5, 0.75, 0.75)
    target_size_hw: tuple[int, int] = (224, 224)
    crop_margin: int = 4
    crop_threshold_hu: float = -500.0
    crop_cc_downsample: int = 4  # downsample factor used only for the connected-component search (memory safety)
    resize_mode: str = "stretch"
    min_change_ratio: float = 0.05
    # Nothing physical reads below air (-1000 HU); lower values are scanner padding (e.g. -8192 outside the
    # circular field of view on Siemens go.All, whose RescaleIntercept is -8192). Flooring them keeps the cache
    # uniform across scanners. None disables it.
    hu_floor: float | None = -1024.0
    device: str = "cpu"  # "cpu", "cuda", or "auto" (use GPU resampling/resizing if a GPU is available)
    version: str = "m1-v3"

    def __post_init__(self) -> None:
        # `-1024` and `-1024.0` must hash to the same cache fingerprint, whichever way the YAML spells it
        if self.hu_floor is not None:
            self.hu_floor = float(self.hu_floor)
