"""Phase 1 (run once per scan): the full pipeline from a raw NIfTI file to a
cached, unwindowed .npy.

    load (real HU, RAS+)  ->  spacing  ->  crop  ->  resize  ->  save int16 .npy

Windowing is NOT part of this package at all: the saved .npy holds raw HU, and
the model code that consumes the cache applies whatever windows it wants, so
they can change without re-running this whole (expensive) step.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .loaders import source_signature
from .pipeline import process_scan
from .preprocess_config import PreprocessConfig
from .quality import QCThresholds

__all__ = [
    "PreprocessConfig", "PreprocessResult", "preprocess_one", "config_fingerprint",
    "is_cache_fresh", "load_cached_stats",
]


@dataclass
class PreprocessResult:
    volume_id: str
    ok: bool
    n_slices: int = 0
    out_shape: tuple[int, ...] | None = None
    out_path: str | None = None
    seconds: float = 0.0
    error: str | None = None
    hu_check: dict | None = None
    spacing_after_resample: tuple[float, float, float] | None = None
    crop_shape: tuple[int, ...] | None = None
    qc_passed: bool | None = None


def config_fingerprint(cfg: PreprocessConfig) -> str:
    """A short, stable hash of every setting that affects the cached .npy.

    Used to detect a stale cache: if a volume's .npy was produced under a
    different config (e.g. before switching CT-RATE download folders, or
    after any preprocessing setting changed), it must NOT be silently reused
    -- see docs/preprocessing/data_contract.md for the bug this fixes.

    ``device`` is deliberately left out: running the same settings on a CPU or
    a GPU produces the same cache, so switching devices must not make every
    cached volume look stale (raw data is gone after ingest, so a false "stale"
    could not even be repaired).
    """
    settings = asdict(cfg)
    settings.pop("device", None)
    payload = json.dumps(settings, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def preprocess_one(
    scan_path: str | Path,
    out_dir: str | Path,
    cfg: PreprocessConfig,
    volume_id: str | None = None,
    rescale_slope: float | None = None,
    rescale_intercept: float | None = None,
    scan_format: str | None = None,
    qc_thresholds: QCThresholds | None = None,
    series_uid: str | None = None,
) -> PreprocessResult:
    """Run the full Phase-1 pipeline on one scan (NIfTI file or DICOM folder)
    and save it to disk.

    This is a thin, cache-side wrapper around pipeline.process_scan() -- the
    shared core that inference (inference.run_inference) also calls, so the
    two can never process a scan differently. This function's own job is only:
    persist the result to disk, always (regardless of QC outcome -- exclusion
    happens later, in scripts/preprocessing/qc_report.py and assign_splits.py,
    not here), and write a sidecar (config fingerprint, raw-input signature,
    per-volume stats) so a later run can tell a stale cache from a fresh one.

    Never raises on a bad input file -- failures are captured and returned as
    a result with ``ok=False`` and an ``error`` message, so a batch run over
    thousands of scans doesn't die on the first broken file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vid = volume_id or Path(scan_path).name.split(".")[0]

    result = process_scan(
        scan_path, cfg, volume_id=vid, rescale_slope=rescale_slope, rescale_intercept=rescale_intercept,
        qc_thresholds=qc_thresholds, scan_format=scan_format, series_uid=series_uid,
    )
    if result.error is not None or result.hu is None:
        return PreprocessResult(volume_id=vid, ok=False, seconds=time.time() - t0, error=result.error)

    out = result.hu.astype(np.int16)  # HU fits comfortably in int16; saved UNwindowed
    out_path = out_dir / f"{vid}.npy"
    np.save(out_path, out)

    fingerprint = config_fingerprint(cfg)
    sz, sy, sx = result.spacing_after_resample
    stats = {
        "n_slices": int(out.shape[0]),
        "spacing_z_mm": sz,
        "spacing_y_mm": sy,
        "spacing_x_mm": sx,
        "crop_shape": "x".join(map(str, result.crop_shape)),
        "qc_passed": bool(result.qc.passed) if result.qc is not None else None,
    }
    meta_path = out_dir / f"{vid}.meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "version": cfg.version,
                # the raw input's identity, so a later run notices if the RAW data changed
                "source_signature": source_signature(scan_path, scan_format, series_uid=series_uid),
                # geometry facts only the loader can know; qc_report.py re-applies them
                "loader_checks": {
                    k: result.meta[k] for k in ("z_uniform", "slice_axis_tilt_deg") if result.meta and k in result.meta
                },
                # per-volume numbers the manifest carries; kept here too so a chunk
                # resumed after a crash can rebuild them without reprocessing
                "stats": stats,
            }
        )
    )

    return PreprocessResult(
        volume_id=vid,
        ok=True,
        n_slices=int(out.shape[0]),
        out_shape=tuple(out.shape),
        out_path=str(out_path),
        seconds=time.time() - t0,
        spacing_after_resample=result.spacing_after_resample,
        crop_shape=result.crop_shape,
        qc_passed=result.qc.passed if result.qc is not None else None,
    )


def load_cached_stats(cache_dir: str | Path, volume_id: str) -> dict | None:
    """The per-volume stats preprocess_one recorded next to the cache file
    (n_slices, spacing_*_mm, crop_shape, qc_passed, npy_path), or None if the
    sidecar is missing/unreadable or predates this field."""
    cache_dir = Path(cache_dir)
    try:
        stats = json.loads((cache_dir / f"{volume_id}.meta.json").read_text()).get("stats")
    except (json.JSONDecodeError, OSError):
        return None
    if not stats:
        return None
    return {**stats, "npy_path": str(cache_dir / f"{volume_id}.npy")}


def is_cache_fresh(
    cache_dir: str | Path,
    volume_id: str,
    cfg: PreprocessConfig,
    scan_path: str | Path | None = None,
    scan_format: str | None = None,
    series_uid: str | None = None,
) -> bool:
    """True only if a cached .npy exists AND is still valid for this run.

    It is valid only if it was produced by this exact config, and -- when the
    raw ``scan_path`` is given -- from raw data of the same size (file count
    and bytes). Anything else (missing file, unreadable sidecar, config
    mismatch, raw data changed) means "needs reprocessing".

    The raw-data check matters: re-downloading the same volume ids from a
    different CT-RATE folder (train -> train_fixed) leaves the config
    untouched, so a config fingerprint alone would happily reuse the old,
    differently-calibrated result.
    """
    cache_dir = Path(cache_dir)
    npy_path = cache_dir / f"{volume_id}.npy"
    meta_path = cache_dir / f"{volume_id}.meta.json"
    if not npy_path.exists() or not meta_path.exists():
        return False
    try:
        saved = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if saved.get("fingerprint") != config_fingerprint(cfg):
        return False
    if scan_path is not None:
        try:
            return saved.get("source_signature") == source_signature(scan_path, scan_format, series_uid=series_uid)
        except (OSError, ValueError):
            return False
    return True
