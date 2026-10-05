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

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DEFAULT_CONFIG_PATH, DataConfig, load_config
from .dicom_loader import discover_scans, make_scan_id
from .loaders import detect_format
from .pipeline import process_scan
from .preprocess import config_fingerprint
from .preprocess_config import PreprocessConfig
from .quality import QCResult, QCThresholds


class ConfigMismatch(ValueError):
    """The preprocessing settings in use differ from the ones the cache was built with."""


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


def check_matches_cache(cfg: PreprocessConfig, data_cfg: DataConfig) -> None:
    """Raise ConfigMismatch if ``cfg`` is not what built the cache.

    ``merge_manifests.py`` records the cache's settings fingerprint in
    ``preprocessing_manifest.json``. A model trained on that cache expects
    inputs prepared exactly that way, so inference with other settings (a
    different size, spacing, crop or HU floor) would feed it differently
    prepared images with no error anywhere. No record yet (no cache has
    been merged on this machine) means nothing to compare, so no check.
    """
    record_path = Path(data_cfg.paths.preprocessing_manifest_path)
    if not record_path.is_file():
        return
    recorded = json.loads(record_path.read_text(encoding="utf-8")).get("fingerprint")
    if recorded and recorded != config_fingerprint(cfg):
        raise ConfigMismatch(
            f"the preprocessing settings differ from the ones the cache in {record_path} was built with "
            f"(fingerprint {config_fingerprint(cfg)} vs {recorded}). Restore the settings in the config, "
            "re-ingest, or pass check_cache=False if the difference is intended."
        )


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
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    check_cache: bool = True,
) -> list[InferenceResult]:
    """Turn one request -- one scan, or a folder of several -- into a
    per-scan result list, ready to feed to a model one row at a time.

    ``rescale_slope``/``rescale_intercept`` only matter for a NIfTI file
    without its own calibration metadata; a DICOM series always carries its
    own values per slice. ``scan_format`` overrides auto-detection for the
    whole request.

    With no ``cfg``, the settings come from ``config_path`` (by default the
    same configs/preprocessing.yaml the cache was built with), including its
    QC thresholds, and are checked against the cache's recorded fingerprint
    (``check_cache``): a YAML edited after the cache was built raises
    ConfigMismatch instead of quietly preparing scans differently. The default
    path is relative to the working directory, like every script's; a service
    started elsewhere should pass an absolute ``config_path``. A ``cfg`` passed
    explicitly is used as given and not checked -- call ``check_matches_cache``
    yourself if that matters.
    """
    if cfg is None:
        try:
            data_cfg = load_config(config_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"cannot find the preprocessing config {config_path} (run from the repository root, "
                "or pass config_path= or cfg=)"
            ) from exc
        cfg = data_cfg.preprocess
        qc_thresholds = qc_thresholds or data_cfg.qc
        if check_cache:
            check_matches_cache(cfg, data_cfg)
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
