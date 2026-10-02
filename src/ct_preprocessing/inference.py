"""The clinical/inference front door -- the second thing that calls
ct_preprocessing.pipeline.process_scan(), alongside the training batch path
(ct_preprocessing.preprocess.preprocess_one). Both call the exact same core, so a
scan is never processed differently at inference than it was at training
time (see docs/preprocessing/data_contract.md, "training-serving skew").

What's genuinely different here, and only here:
  - ``path`` can point at one scan (a NIfTI file, or a single DICOM series
    folder) or at a folder holding many scans (many patients/series) --
    discovered the same tag-based way training data is, via
    ct_preprocessing.dicom_loader.discover_scans. One request can mean one scan
    or a small batch; both go through the same code path here.
  - No labels, and no split column -- a real request has none; producing a
    label is the model's job (M4), and there is no train/val/test to assign
    for a one-off prediction request.
  - Nothing is written to disk: no manifest, no persistent cache, no
    montages. Every result is transient, held only for the caller.
  - One bad scan in a batch never stops the rest: a hard error or a QC
    failure is recorded on that scan's own row (passed=False, no tensor),
    and the loop moves on -- the same log-and-continue shape
    scripts/preprocess_all.py uses for training, not the old
    raise-and-stop-everything behaviour this module used to have. The
    caller (a real clinical UI, or M4) is expected to check ``passed`` per
    row and refuse to show a prediction for any row where it's False.

This module intentionally stops at "here is a ready tensor per scan, or here
is why a given scan was rejected" -- feeding a tensor to the actual
classification model is M4's job, once it exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import slices_to_tensor
from .dicom_loader import discover_scans, make_scan_id
from .loaders import detect_format
from .pipeline import process_scan
from .preprocess_config import PreprocessConfig
from .quality import QCResult, QCThresholds


@dataclass
class InferenceResult:
    volume_id: str
    scan_path: str  # what was actually opened -- a NIfTI file or a DICOM series folder
    format: str
    passed: bool  # False on a hard error OR a failed QC check -- either way, no tensor
    pixel_values: np.ndarray | None  # (K, 3, H, W) float32 in [0, 1]; None unless passed
    qc: QCResult | None  # None only on a hard error (bad/unreadable file)
    error: str | None  # set only on a hard error


def _discover_for_inference(path: Path, fmt: str | None) -> list[tuple[Path, str, str | None]]:
    """(scan_path, format, series_uid) for every scan found at/under ``path``.

    A single file is always one NIfTI scan. A directory may itself be one
    DICOM series, or hold many scans -- discover_scans handles both without
    this function needing to guess which.
    """
    if path.is_file():
        return [(path, fmt or detect_format(path), None)]
    entries = discover_scans(path, fmt=fmt or "auto")
    if not entries:
        raise FileNotFoundError(f"no scans found at or under {path}")
    return [(path if e.scan_path == "." else path / e.scan_path, e.format, e.series_uid) for e in entries]


def run_inference(
    path: str | Path,
    cfg: PreprocessConfig | None = None,
    qc_thresholds: QCThresholds | None = None,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
    windows: dict | None = None,
    normalize_imagenet: bool = True,
    scan_format: str | None = None,
) -> list[InferenceResult]:
    """Turn one request -- one scan, or a folder of several -- into a
    per-scan result table, ready to feed to a model one row at a time.

    ``rescale_slope``/``rescale_intercept`` only matter for a NIfTI file
    without its own calibration metadata; a DICOM series always carries its
    own values per slice. ``scan_format`` overrides auto-detection for the
    whole request.
    """
    cfg = cfg or PreprocessConfig()
    path = Path(path)
    scans = _discover_for_inference(path, scan_format)

    results: list[InferenceResult] = []
    for scan_path, fmt, series_uid in scans:
        rel = scan_path.relative_to(path) if scan_path != path else Path(".")
        vid = make_scan_id(rel.as_posix(), series_uid) if fmt == "dicom" else scan_path.name.split(".")[0]

        result = process_scan(
            scan_path, cfg, volume_id=vid, rescale_slope=rescale_slope, rescale_intercept=rescale_intercept,
            qc_thresholds=qc_thresholds, scan_format=fmt, series_uid=series_uid,
        )

        if result.error is not None or result.hu is None:
            results.append(InferenceResult(
                volume_id=vid, scan_path=str(scan_path), format=fmt, passed=False,
                pixel_values=None, qc=None, error=result.error,
            ))
            continue

        if not result.qc.passed:
            results.append(InferenceResult(
                volume_id=vid, scan_path=str(scan_path), format=fmt, passed=False,
                pixel_values=None, qc=result.qc, error=None,
            ))
            continue

        pixel_values = slices_to_tensor(result.hu, windows=windows, normalize_imagenet=normalize_imagenet)
        results.append(InferenceResult(
            volume_id=vid, scan_path=str(scan_path), format=fmt, passed=True,
            pixel_values=pixel_values, qc=result.qc, error=None,
        ))

    return results
