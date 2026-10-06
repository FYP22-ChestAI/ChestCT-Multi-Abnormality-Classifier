"""The one place "load -> calibrate -> resample -> crop -> resize -> QC" is
sequenced. Both the training batch pipeline (preprocess.py, which persists
the result to disk) and the single-scan inference entry point (inference.py)
call this SAME function -- so the two can never quietly drift apart and
process a scan differently (see docs/preprocessing/data_contract.md, "training-serving
skew"). It works on either file format: the loader is chosen by
loaders.load_volume, and everything after it sees only a ``Volume``.

process_scan() never decides what to do about a QC failure -- it always
returns the processed array plus the QCResult, and leaves that decision to
the caller. Training keeps saving every volume and lets qc_report.py
exclude failures later; the inference caller checks `result.qc.passed`
itself and refuses to proceed to the model on failure. Data that cannot be
stored at all (NaN/Inf, or HU outside the int16 range) is not a QC question:
it is a hard error, with no array returned.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .crop import crop_to_foreground
from .loader import ensure_calibrated_hu
from .loaders import load_volume
from .preprocess_config import PreprocessConfig
from .quality import QCResult, QCThresholds, check_volume, int16_problem, nonfinite_problem
from .resize import resize_slices
from .spacing import resample_to_spacing
from .types import Volume

MAX_SLICE_AXIS_TILT_DEG = 1.0


@dataclass
class ScanResult:
    volume_id: str
    hu: np.ndarray | None  # (N, H, W) int16, unwindowed -- None only on a hard error
    qc: QCResult | None  # None only on a hard error
    spacing_after_resample: tuple[float, float, float] | None = None
    crop_shape: tuple[int, ...] | None = None
    calibration: dict | None = None
    meta: dict | None = None  # what the loader recorded (format, acquisition fields, ...)
    error: str | None = None


def apply_loader_checks(qc: QCResult, meta: dict) -> None:
    """DICOM-specific geometry problems the array alone can't reveal.

    Uneven slice spacing means resampling would distort the anatomy (a
    missing slice, a gap), so it fails QC. A tilted slice axis (gantry tilt)
    is only flagged: it is rare and the data is still usable, but a human
    should know.
    """
    if meta.get("z_uniform") is False:
        qc.reasons.append("non-uniform slice spacing (missing or duplicated slices?) -- resampling would distort it")
        qc.passed = False
    tilt = meta.get("slice_axis_tilt_deg")
    if tilt is not None and tilt > MAX_SLICE_AXIS_TILT_DEG:
        qc.flags.append(f"slice axis tilted {tilt:.1f} deg from the slice normal (gantry tilt?)")


def _floor_padding(hu: np.ndarray, floor: float | None, calibration: dict) -> np.ndarray:
    """Set every value below ``floor`` to ``floor`` and record what was there.

    Scanner padding outside the circular field of view reads -1024 on most
    scanners but -8192 on Siemens go.All (RescaleIntercept -8192). Nothing
    real is below air, so flooring makes the cache uniform; lungs, bone and
    metal are untouched. Done before resampling so the interpolation never
    blends -8192 into the edge of the body. The raw HU range seen is stored
    in the calibration record, since the raw file is deleted after ingest.
    """
    vmin, vmax = float(hu.min()), float(hu.max())
    calibration["hu_range_before_floor"] = [vmin, vmax]
    calibration["hu_floor"] = floor
    calibration["floored"] = bool(floor is not None and vmin < floor)
    if not calibration["floored"]:
        return hu
    if hu.flags.writeable:
        np.maximum(hu, floor, out=hu)  # in place: a second copy of a large volume is real memory
        return hu
    return np.maximum(hu, floor)


def process_scan(
    scan_path: str | Path,
    cfg: PreprocessConfig,
    volume_id: str | None = None,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
    qc_thresholds: QCThresholds | None = None,
    scan_format: str | None = None,
    series_uid: str | None = None,
) -> ScanResult:
    """Run the full pipeline on one scan and return the result in memory --
    nothing is written to disk here (that's preprocess.py's job for the
    training path; the inference entry point feeds the returned array
    straight to the model instead).

    ``scan_format`` is "nifti", "dicom", or None to detect it from the path.
    ``rescale_slope``/``rescale_intercept`` matter only for NIfTI (CT-RATE's
    metadata CSV); a DICOM loader always applies each slice's own values.
    ``series_uid`` picks out one series when ``scan_path``'s folder holds
    more than one (from discover_scans) -- unused for NIfTI.
    """
    vid = volume_id or Path(scan_path).name.split(".")[0]

    try:
        volume = load_volume(scan_path, fmt=scan_format, volume_id=vid, series_uid=series_uid)
        loader_meta = dict(volume.meta)

        problem = nonfinite_problem(volume.hu)  # a NaN would spread through the interpolation below
        if problem:
            return ScanResult(volume_id=vid, hu=None, qc=None, error=problem)

        hu, calib = ensure_calibrated_hu(volume.hu, rescale_slope, rescale_intercept)
        hu = _floor_padding(hu, cfg.hu_floor, calib)
        volume = Volume(hu=hu, spacing=volume.spacing, orientation=volume.orientation, meta=volume.meta)

        volume = resample_to_spacing(volume, cfg.target_spacing_zyx, cfg.min_change_ratio, device=cfg.device)
        spacing_after_resample = volume.spacing

        volume = crop_to_foreground(
            volume, threshold_hu=cfg.crop_threshold_hu, margin=cfg.crop_margin, cc_downsample=cfg.crop_cc_downsample
        )
        crop_shape = tuple(volume.hu.shape)

        resized = resize_slices(volume.hu, cfg.target_size_hw, mode=cfg.resize_mode, device=cfg.device)
        resized = np.rint(resized)  # astype() would truncate toward zero (40.9 -> 40, -0.9 -> 0)
        problem = int16_problem(resized)
        if problem:  # never cast first: NaN/Inf become 0 and 40000 wraps to -25536, and QC would pass both
            return ScanResult(volume_id=vid, hu=None, qc=None, error=problem)
        out = resized.astype(np.int16)

        qc = check_volume(vid, out, spacing_after_resample, qc_thresholds)
        apply_loader_checks(qc, loader_meta)

        return ScanResult(
            volume_id=vid,
            hu=out,
            qc=qc,
            spacing_after_resample=spacing_after_resample,
            crop_shape=crop_shape,
            calibration=calib,
            meta=loader_meta,
        )
    except Exception as exc:  # noqa: BLE001 - every failure is captured, never raised, for a batch run
        return ScanResult(volume_id=vid, hu=None, qc=None, error=str(exc))
