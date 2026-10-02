"""Combine the per-chunk manifests into the one manifest.csv the model code reads.

The merged manifest has one row per volume, deduplicated by volume_id (an
archive uploaded twice would otherwise appear twice). Splits already frozen by
assign_splits.py are joined back in, so re-merging after a top-up never loses
them; patients with no frozen assignment yet are "unassigned".
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..config import DataConfig
from ..preprocess import config_fingerprint
from . import state
from .splits import apply_splits, read_splits


@dataclass
class MergeReport:
    rows_per_source: dict[str, int] = field(default_factory=dict)
    duplicates_dropped: int = 0
    failures: list[dict] = field(default_factory=list)  # volumes that could not be processed


def read_chunk_manifests(cfg: DataConfig, source_name: str) -> pd.DataFrame:
    folder = state.chunk_manifest_dir(cfg.paths, source_name)
    files = sorted(folder.glob("chunk_*.csv")) if folder.is_dir() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, dtype={"ingest_chunk": str}) for f in files], ignore_index=True)


def collect_failures(cfg: DataConfig, source_name: str) -> list[dict]:
    out = []
    for chunk_id, info in state.done_chunks(cfg.paths, source_name).items():
        for volume_id, error in (info.get("failed") or {}).items():
            out.append({"source_name": source_name, "ingest_chunk": chunk_id, "volume_id": volume_id, "error": error})
    return out


def merge_sources(cfg: DataConfig, source_names: list[str]) -> tuple[pd.DataFrame, MergeReport]:
    report = MergeReport()
    frames = []
    for name in source_names:
        df = read_chunk_manifests(cfg, name)
        if df.empty:
            continue
        before = len(df)
        df = df.drop_duplicates("volume_id", keep="first")
        report.duplicates_dropped += before - len(df)
        report.rows_per_source[name] = len(df)
        report.failures += collect_failures(cfg, name)
        frames.append(df)
    if not frames:
        raise ValueError("no chunk manifests found -- run ingest.py first")
    merged = pd.concat(frames, ignore_index=True)
    dup = merged["volume_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"volume ids are not unique across sources: {merged.loc[dup, 'volume_id'].tolist()[:6]}")
    frozen = {name: read_splits(state.splits_path(cfg.paths, name)) for name in report.rows_per_source}
    return apply_splits(merged, frozen), report


def read_patients_file(path: str | Path) -> list[str]:
    """One patient folder name per line; blank lines and '#' comments are ignored."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip().rstrip("/\\")
        if line and not line.startswith("#"):
            out.append(line)
    return out


def reconcile_patients(rows: pd.DataFrame, expected: list[str]) -> tuple[list[str], list[str]]:
    """(missing, unexpected): patient folders on the source disk that never made
    it into the manifest, and manifest folders that were not on the checklist.
    Compares the top-level folder of each scan_path, so it works whichever way
    patients were grouped."""
    found = set(rows["scan_path"].astype(str).str.split("/").str[0])
    return sorted(set(expected) - found), sorted(found - set(expected))


def suspicious_patients(rows: pd.DataFrame, *, min_scans: int = 20, factor: float = 10.0) -> list[tuple[str, int]]:
    """Patients with implausibly many scans compared with the rest -- the classic sign
    of an archive zipped one folder too high, which would turn a whole cohort into a
    single giant "patient" (and make patient-level splitting meaningless).
    Returns (patient_id, n_scans), largest first."""
    counts = rows.groupby("patient_id").size()
    if counts.empty:
        return []
    limit = max(min_scans, factor * counts.median())
    return [(p, int(n)) for p, n in counts[counts > limit].sort_values(ascending=False).items()]


def write_run_record(path: str | Path, cfg: DataConfig, manifest: pd.DataFrame, report: MergeReport) -> None:
    """The settings that produced the cache, plus what failed -- a reproducibility
    record (and a fingerprint the model code can compare to know its cache is current)."""
    record = {
        "preprocess_config": asdict(cfg.preprocess),
        "fingerprint": config_fingerprint(cfg.preprocess),
        "n_volumes": int(len(manifest)),
        "volumes_per_source": report.rows_per_source,
        "n_failed_volumes": len(report.failures),
        "failed": report.failures,
    }
    path = Path(path)
    state.write_atomic(path, json.dumps(record, indent=2, default=str))
