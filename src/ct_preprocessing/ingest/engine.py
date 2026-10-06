"""The ingest loop: for every pending chunk, fetch -> manifest rows ->
preprocess into the permanent cache -> delete the raw data -> mark done.

Designed to run unattended for days inside tmux (or a batch job):

  * resumable -- a chunk is only skipped once its ``.done`` marker exists, and
    a half-processed chunk resumes without re-fetching volumes already cached;
  * never fills the disk -- it stops cleanly when free space drops below
    ``min_free_gb`` instead of crashing the server;
  * survives bad data -- a failed volume or chunk is recorded and the run moves
    on; only a run of consecutive failed chunks (a network or login problem)
    stops it;
  * raw data never accumulates -- scratch is emptied after every chunk.
"""
from __future__ import annotations

import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from ..cache_record import ensure_cache_matches
from ..config import DataConfig
from ..manifest import FOLDER_BUILDER
from ..preprocess import PreprocessConfig, is_cache_fresh, load_cached_stats, preprocess_one
from ..quality import QCThresholds
from ..runs import Run, list_runs
from . import state
from .base import FetchError, Fetcher
from .lock import SourceLock


@dataclass
class IngestOptions:
    max_chunks: int | None = None
    only_chunk: str | None = None
    dry_run: bool = False
    retry_failed: bool = False
    min_free_gb: float = 100.0
    workers: int = 4
    max_consecutive_failures: int = 3
    allow_new_settings: bool = False  # let new preprocessing settings replace the cache's recorded ones (old volumes go stale)


@dataclass
class IngestSummary:
    chunks_done: int = 0
    chunks_failed: int = 0
    volumes_ok: int = 0
    volumes_failed: int = 0
    pending_after: int = 0
    stopped: str | None = None  # why the run ended early, if it did
    seconds: float = 0.0


# ----------------------------------------------------------------- preprocessing
def _run_one(job: tuple):
    scan_path, out_dir, cfg, volume_id, slope, intercept, scan_format, qc, series_uid = job
    return preprocess_one(
        scan_path, out_dir, cfg, volume_id=volume_id, rescale_slope=slope, rescale_intercept=intercept,
        scan_format=scan_format, qc_thresholds=qc, series_uid=series_uid,
    )


def _first_present(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return None


@dataclass
class RowsResult:
    stats: pd.DataFrame
    failed: dict[str, str] = field(default_factory=dict)
    n_ok: int = 0
    n_cached: int = 0


def preprocess_rows(
    rows: pd.DataFrame,
    *,
    raw_dir: Path,
    cache_dir: Path,
    preprocess_cfg: PreprocessConfig,
    qc: QCThresholds,
    workers: int = 1,
    force: bool = False,
) -> RowsResult:
    """Run the shared core over manifest rows whose raw files are under ``raw_dir``.

    A volume with a fresh cache entry is skipped -- even when its raw file is
    already gone, which is the normal state after an earlier chunk was cleaned
    up. A volume with no raw file and no fresh cache is a recorded failure, not
    a crash. Returns the per-volume stats (so the manifest carries them) and
    the failures.
    """
    raw_dir, cache_dir = Path(raw_dir), Path(cache_dir)
    jobs, failed, stats, n_cached = [], {}, [], 0
    for _, row in rows.iterrows():
        vid, fmt = row["volume_id"], row["format"]
        scan_path = raw_dir / row["scan_path"]
        series_uid = row["series_uid"] if "series_uid" in row and pd.notna(row["series_uid"]) else None
        raw_exists = scan_path.exists()
        if not force:
            fresh = (
                is_cache_fresh(cache_dir, vid, preprocess_cfg, scan_path=scan_path, scan_format=fmt, series_uid=series_uid)
                if raw_exists
                else is_cache_fresh(cache_dir, vid, preprocess_cfg)
            )
            cached = load_cached_stats(cache_dir, vid) if fresh else None
            if cached is not None:
                stats.append({"volume_id": vid, **cached})
                n_cached += 1
                continue
        if not raw_exists:
            failed[vid] = "raw file missing and no fresh cache entry"
            continue
        jobs.append((
            str(scan_path), str(cache_dir), preprocess_cfg, vid,
            _first_present(row, ["RescaleSlope"]), _first_present(row, ["RescaleIntercept"]), fmt, qc, series_uid,
        ))

    if workers <= 1 or len(jobs) <= 1:
        results = [_run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_run_one, jobs))

    n_ok = 0
    for r in results:
        if not r.ok:
            failed[r.volume_id] = r.error or "unknown error"
            continue
        n_ok += 1
        sz, sy, sx = r.spacing_after_resample
        stats.append({
            "volume_id": r.volume_id, "n_slices": r.n_slices, "npy_path": r.out_path,
            "spacing_z_mm": sz, "spacing_y_mm": sy, "spacing_x_mm": sx,
            "crop_shape": "x".join(map(str, r.crop_shape)), "qc_passed": r.qc_passed,
        })
    return RowsResult(stats=pd.DataFrame(stats), failed=failed, n_ok=n_ok, n_cached=n_cached)


# ------------------------------------------------------------------------ helpers
def free_gb(*folders: Path) -> float:
    """Free space, in GB, on the fullest of the given folders' filesystems."""
    values = []
    for f in folders:
        f = Path(f)
        f.mkdir(parents=True, exist_ok=True)
        values.append(shutil.disk_usage(f).free / 1e9)
    return min(values)


def _cache_gb(volume_ids, cache_dir: Path) -> float:
    total = 0
    for v in volume_ids:
        try:
            total += os.path.getsize(cache_dir / f"{v}.npy")
        except OSError:
            pass
    return total / 1e9


def resolve_patient_id_source(paths, source_name: str, configured: str) -> str:
    """An explicit config value wins; "auto" sticks with the first decision made
    for this source, so every chunk groups patients the same way."""
    if configured != "auto":
        return configured
    return state.read_patient_id_source(paths, source_name) or "auto"


# --------------------------------------------------------------------------- loop
def run_ingest(
    run: Run,
    cfg: DataConfig,
    fetcher: Fetcher,
    options: IngestOptions,
    *,
    log: Callable[[str], None] = print,
) -> IngestSummary:
    """Ingest every pending chunk of ``run``. One ingest per SOURCE at a time (all of a source's runs
    share its scratch folder; see lock.py); a dry run only reports and needs no lock. The cache is
    shared by every run, so volumes another run already cached are not fetched again."""
    if options.dry_run:
        return _ingest(run, cfg, fetcher, options, log=log)
    with SourceLock(state.source_state_dir(cfg.paths, run.source) / "ingest.lock", run.source):
        ensure_cache_matches(cfg.paths.cache_dir, cfg.preprocess, allow_new_settings=options.allow_new_settings)
        return _ingest(run, cfg, fetcher, options, log=log)


def _ingest(
    run: Run,
    cfg: DataConfig,
    fetcher: Fetcher,
    options: IngestOptions,
    *,
    log: Callable[[str], None] = print,
) -> IngestSummary:
    source_name = run.source
    source_cfg = cfg.source(source_name)
    paths = cfg.paths
    raw_dir, cache_dir = Path(source_cfg.raw_dir), Path(paths.cache_dir)
    qc = cfg.qc_for(source_name)
    summary = IngestSummary()
    t_start = time.time()

    done = state.done_chunks(run)
    if options.retry_failed:
        with_failures = [c for c, info in done.items() if info.get("n_failed", 0) > 0]
        if not options.dry_run:
            for chunk_id in with_failures:
                state.clear_done(run, chunk_id)
        done = {c: i for c, i in done.items() if c not in with_failures}

    all_chunks = fetcher.chunk_ids()
    pending = [c for c in all_chunks if c not in done]
    if options.only_chunk is not None:
        if options.only_chunk not in all_chunks:
            raise ValueError(f"chunk {options.only_chunk!r} is not one of this source's chunks")
        pending = [c for c in pending if c == options.only_chunk]
    log(f"[{source_name}/{run.name}] {len(pending)} chunk(s) pending of {len(all_chunks)} ({len(done)} already done)")
    if options.dry_run:
        for c in pending[: (options.max_chunks or len(pending))]:
            log(f"  would ingest chunk {c}")
        summary.pending_after = len(pending)
        return summary

    state.prepare_scratch(raw_dir)
    consecutive_failures = 0
    for index, chunk_id in enumerate(pending, 1):
        if options.max_chunks is not None and summary.chunks_done + summary.chunks_failed >= options.max_chunks:
            summary.stopped = f"--max-chunks {options.max_chunks} reached"
            break
        free = free_gb(raw_dir, cache_dir)
        if free < options.min_free_gb:
            summary.stopped = f"free disk {free:.0f} GB is below min_free_gb {options.min_free_gb:g} GB"
            break
        if consecutive_failures >= options.max_consecutive_failures:
            summary.stopped = f"{consecutive_failures} chunks in a row failed -- check the network / logins"
            break

        t0 = time.time()
        tag = f"[chunk {chunk_id}] ({index}/{len(pending)})"
        try:
            state.clear_scratch(raw_dir)
            report = fetcher.fetch(
                chunk_id, raw_dir, cache_fresh=lambda v: is_cache_fresh(cache_dir, v, cfg.preprocess)
            )
            pid_source = resolve_patient_id_source(paths, source_name, source_cfg.patient_id_source)
            rows = fetcher.build_rows(chunk_id, raw_dir, patient_id_source=pid_source)
            if source_cfg.manifest_builder == FOLDER_BUILDER and pid_source == "auto":
                state.write_patient_id_source(paths, source_name, rows.attrs.get("patient_id_source", "path"))
            result = preprocess_rows(
                rows, raw_dir=raw_dir, cache_dir=cache_dir, preprocess_cfg=cfg.preprocess, qc=qc, workers=options.workers
            )
            failed = {**result.failed, **{v: f"download failed: {e}" for v, e in report.failed.items()}}
            rows = rows.assign(source_name=source_name, ingest_chunk=chunk_id)
            if not result.stats.empty:
                rows = rows.merge(result.stats, on="volume_id", how="left")
            manifest_file = run.chunk_manifest_path(chunk_id)
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            rows.to_csv(manifest_file, index=False)
            state.write_done(run, chunk_id, {
                "n_volumes": len(rows), "n_ok": result.n_ok, "n_cached": result.n_cached,
                "n_failed": len(failed), "failed": failed, "seconds": round(time.time() - t0, 1),
            })
            summary.chunks_done += 1
            summary.volumes_ok += result.n_ok + result.n_cached
            summary.volumes_failed += len(failed)
            consecutive_failures = 0
            log(
                f"{tag} {len(rows)} volumes: {result.n_ok} ok, {result.n_cached} already cached, {len(failed)} failed | "
                f"cache +{_cache_gb(rows['volume_id'], cache_dir):.1f} GB | free {free_gb(raw_dir, cache_dir):.0f} GB | "
                f"{time.time() - t0:.0f}s"
            )
        except (FetchError, ValueError, OSError, RuntimeError) as exc:
            state.write_failed(run, chunk_id, f"{type(exc).__name__}: {exc}")
            summary.chunks_failed += 1
            consecutive_failures += 1
            log(f"{tag} FAILED: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 - one unexpected error must not end a multi-day unattended run
            state.write_failed(run, chunk_id, traceback.format_exc(limit=3))
            summary.chunks_failed += 1
            consecutive_failures += 1
            log(f"{tag} UNEXPECTED ERROR: {type(exc).__name__}: {exc}")
        finally:
            state.clear_scratch(raw_dir)


    summary.pending_after = len([c for c in all_chunks if c not in state.done_chunks(run)])
    summary.seconds = time.time() - t_start
    log(
        f"[{source_name}/{run.name}] finished: {summary.chunks_done} chunk(s) done, {summary.chunks_failed} failed, "
        f"{summary.volumes_ok} volumes ok, {summary.volumes_failed} failed volumes, {summary.pending_after} chunk(s) still pending"
        + (f" -- stopped: {summary.stopped}" if summary.stopped else "")
    )
    return summary


def _run_cache_gb(run: Run, cache_dir: Path) -> tuple[float, int]:
    """(GB, volumes) of the cache files this run lists, from its chunk manifests."""
    folder = run.chunk_manifest_dir
    ids: list[str] = []
    if folder.is_dir():
        for f in sorted(folder.glob("chunk_*.csv")):
            ids += pd.read_csv(f, usecols=["volume_id"])["volume_id"].tolist()
    return _cache_gb(ids, cache_dir), len(ids)


def ingest_status(run: Run, cfg: DataConfig, fetcher: Fetcher | None = None) -> str:
    """A short progress report for one run: chunks done, failures, cache size, free disk; plus a line per
    other run of the same source."""
    paths = cfg.paths
    done = state.done_chunks(run)
    failed = state.failed_chunks(run)
    cache_dir = Path(paths.cache_dir)
    n_vol = sum(i.get("n_ok", 0) + i.get("n_cached", 0) for i in done.values())
    n_bad = sum(i.get("n_failed", 0) for i in done.values())
    total = len(fetcher.chunk_ids()) if fetcher is not None else None
    gb, n_listed = _run_cache_gb(run, cache_dir)
    lines = [
        f"[{run.source}/{run.name}] chunks done: {len(done)}" + (f" / {total}" if total is not None else "")
        + f" | failed chunks: {len(failed)} | volumes ok: {n_vol}, failed: {n_bad}",
        f"cache: {gb:.1f} GB listed by this run | free disk: {free_gb(cache_dir):.0f} GB",
    ]
    if total and done and gb > 0:
        measured_mb = gb * 1000 / max(n_vol, 1)
        projected = measured_mb * (n_listed / len(done)) * total / 1000
        lines.append(f"measured {measured_mb:.1f} MB/volume -> projected ~{projected:.0f} GB at the end of this run")
    for chunk_id, info in list(failed.items())[:5]:
        lines.append(f"  failed chunk {chunk_id} (attempts {info.get('attempts', '?')}): {str(info.get('error', ''))[:120]}")
    others = [name for name in list_runs(paths, run.source) if name != run.name]
    if others:
        lines.append(f"other runs of {run.source}: {', '.join(others)}")
    return "\n".join(lines)
