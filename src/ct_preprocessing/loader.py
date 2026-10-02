"""Loaders turn one raw scan file into a Volume with real Hounsfield Units.

Only a NIfTI loader (for CT-RATE) is implemented here. A DICOM loader for the
local NHRD dataset is separate, later work (see Part B in docs/preprocessing/data_contract.md)
and must return the same Volume shape, so nothing else in this package has to
change when it's added.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .types import Volume

# Real air reads close to -1000 HU and water close to 0 HU. If a loaded
# volume's values sit far outside this, the file's rescale slope/intercept
# probably was not applied -- see check_hu_plausible().
AIR_HU_APPROX = -1000.0
WATER_HU_APPROX = 0.0


def volume_from_nifti_image(img: "nib.spatialimages.SpatialImage", meta: dict) -> Volume:
    """Standardise any nibabel image to our (Z, Y, X) RAS+ convention.

    Both loaders funnel through here -- the NIfTI loader directly, and the
    DICOM loader by first building a nibabel image from the DICOM geometry --
    so the two formats can never end up in different orientation conventions
    (see tests/test_dicom_loader.py, which checks they agree exactly).

    Orientation is standardised to RAS+ using nibabel's own (tested)
    reorientation logic, rather than a hand-rolled transpose.
    """
    img = nib.as_closest_canonical(img)  # standardise orientation to RAS+

    data = np.asarray(img.dataobj).astype(np.float32)  # (X, Y, Z), RAS+ order
    zooms = img.header.get_zooms()[:3]  # (x, y, z) mm, same order as data

    # Our convention for the rest of the pipeline: (Z, Y, X), i.e. slices
    # stack along axis 0. Convenient for "N axial slices" everywhere else.
    hu = np.transpose(data, (2, 1, 0))
    spacing_zyx = (float(zooms[2]), float(zooms[1]), float(zooms[0]))

    return Volume(hu=hu, spacing=spacing_zyx, orientation="RAS+", meta=meta)


def load_nifti_volume(path: str | Path, volume_id: str | None = None) -> Volume:
    """Load a CT-RATE ``.nii.gz`` file into a (Z, Y, X) Hounsfield-Unit Volume."""
    path = Path(path)
    img = nib.load(str(path))
    return volume_from_nifti_image(
        img,
        meta={
            "source_path": str(path),
            "volume_id": volume_id or path.name.split(".")[0],
            "format": "nifti",
        },
    )


def ensure_calibrated_hu(
    hu: np.ndarray,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
    sample_size: int = 200_000,
    uncalibrated_threshold: float = -500.0,
) -> tuple[np.ndarray, dict]:
    """If this volume doesn't look like real HU yet, apply its own
    RescaleSlope/RescaleIntercept (from that scan's metadata) to correct it.

    Real HU always has a low-percentile (background/air) value well below
    -500 -- CT-RATE's own scanner diversity confirmed this can legitimately
    be anywhere from about -1024 to -8192 depending on the scanner, and both
    are fine. What is NOT fine is a low-percentile value near or above 0,
    which is the signature of raw, un-rescaled detector output (the 0-4095
    12-bit range) -- that's the actual defect this function catches and, if
    given this scan's own rescale values, fixes. See docs/preprocessing/data_contract.md.

    Nothing is changed if the volume already looks calibrated, and nothing
    is changed if it looks uncalibrated but no rescale values were given
    (the caller finds out via the returned info dict and can decide what to
    do -- e.g. leave it for the QC step to flag).
    """
    flat = hu.reshape(-1)
    sample = flat if flat.size <= sample_size else np.random.default_rng(0).choice(flat, size=sample_size, replace=False)
    p_low_before = float(np.percentile(sample, 0.5))

    if p_low_before <= uncalibrated_threshold:
        return hu, {"applied": False, "already_calibrated": True, "p_low_before": p_low_before}

    if rescale_slope is None or rescale_intercept is None:
        return hu, {
            "applied": False,
            "already_calibrated": False,
            "p_low_before": p_low_before,
            "reason": "looks uncalibrated but no RescaleSlope/RescaleIntercept was provided",
        }

    corrected = hu * float(rescale_slope) + float(rescale_intercept)
    corrected_sample = corrected.reshape(-1)
    if corrected_sample.size > sample_size:
        corrected_sample = np.random.default_rng(0).choice(corrected_sample, size=sample_size, replace=False)
    p_low_after = float(np.percentile(corrected_sample, 0.5))
    return corrected, {
        "applied": True,
        "already_calibrated": False,
        "p_low_before": p_low_before,
        "p_low_after": p_low_after,
        "rescale_slope": rescale_slope,
        "rescale_intercept": rescale_intercept,
    }


def check_hu_plausible(hu: np.ndarray, sample_size: int = 200_000) -> dict:
    """Report basic stats so a human can judge whether values look like real HU.

    This does not fix anything -- it only reports numbers for the Step 4
    manual check ("open ~10 scans and look at min/max/mean"). For CT-RATE
    this should already be true HU; for a new data source, a min far above
    -500 usually means slope/intercept still needs to be applied.
    """
    flat = hu.reshape(-1)
    if flat.size > sample_size:
        rng = np.random.default_rng(0)
        flat = rng.choice(flat, size=sample_size, replace=False)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "p01": float(np.percentile(flat, 1)),
        "p99": float(np.percentile(flat, 99)),
        "looks_like_hu": bool(np.min(flat) < -500.0),  # real background air must appear
    }
