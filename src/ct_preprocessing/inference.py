"""The inference front door -- the second thing that calls
ct_preprocessing.pipeline.process_scan(), alongside the training ingest path
(ct_preprocessing.ingest.engine). Both call the exact same core, so a scan is
never processed differently at inference than it was at training time (see
docs/preprocessing/data_contract.md, "training-serving skew").

What's genuinely different here, and only here:
  - ``path`` can point at one scan (a NIfTI file, or a single DICOM series
    folder) or at a folder holding many scans (many patients/series) --
    discovered the same tag-based way training data is, via
    ct_preprocessing.dicom_loader.discover_scans. One request can mean one
    scan or a small batch; both go through the same code path here.
  - No labels and no split -- a real request has neither.
  - Nothing is written to disk: no manifest, no cache, no montages. Every
    result is transient, held only for the caller.
  - One bad scan in a batch never stops the rest: a hard error or a QC
    failure is recorded on that scan's own row (passed=False, no volume),
    and the loop moves on. The caller is expected to check ``passed`` per
    row and refuse to use any row where it's False.

The result is the preprocessed volume exactly as the training cache stores it
(raw Hounsfield Units, int16, shape (N, H, W)). Windowing and any other
model-input shaping belong to the model code that consumes it, applied
identically there for training (from the cache) and inference (from here).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    passed: bool  # False on a hard error OR a failed QC check -- either way, no volume
    hu: np.ndarray | None  # (N, H, W) int16 raw HU, same as the cache; None unless passed
    spacing_zyx: tuple[float, float, float] | None  # mm, after resampling; None unless passed
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
    scan_format: str | None = None,
) -> list[InferenceResult]:
    """Turn one request -- one scan, or a folder of several -- into a
    per-scan result list, ready to feed to a model one row at a time.

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
                hu=None, spacing_zyx=None, qc=None, error=result.error,
            ))
        elif not result.qc.passed:
            results.append(InferenceResult(
                volume_id=vid, scan_path=str(scan_path), format=fmt, passed=False,
                hu=None, spacing_zyx=None, qc=result.qc, error=None,
            ))
        else:
            results.append(InferenceResult(
                volume_id=vid, scan_path=str(scan_path), format=fmt, passed=True,
                hu=result.hu, spacing_zyx=result.spacing_after_resample, qc=result.qc, error=None,
            ))

    return results
