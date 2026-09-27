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
    device: str = "cpu"  # "cpu", "cuda", or "auto" (use GPU resampling/resizing if a GPU is available)
    version: str = "m1-v2"
