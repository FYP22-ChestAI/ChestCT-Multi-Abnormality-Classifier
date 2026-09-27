"""Shared data shapes used across the whole M1 pipeline.

Keeping this in one small file means every stage (loader, spacing, crop,
resize, ...) agrees on exactly one definition of what a "volume" is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Volume:
    """One 3D CT volume, in real Hounsfield Units.

    Axis order is always (Z, Y, X) = (slice, row, col) -- i.e. slices stack
    along axis 0. This is the one shape every loader must produce. A future
    DICOM loader (local NHRD data, not implemented yet) must return this same
    shape, so nothing downstream of the loader has to know or care where the
    volume came from.
    """

    hu: np.ndarray  # float32 or int16, shape (n_slices, height, width), real HU
    spacing: tuple[float, float, float]  # (z, y, x) mm per voxel, same axis order as hu
    orientation: str = "RAS+"  # e.g. "RAS+" once standardised; "native" before that
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hu.ndim != 3:
            raise ValueError(f"Volume.hu must be 3D (Z, Y, X), got shape {self.hu.shape}")
        if len(self.spacing) != 3:
            raise ValueError(f"Volume.spacing must be a (z, y, x) triple, got {self.spacing!r}")
