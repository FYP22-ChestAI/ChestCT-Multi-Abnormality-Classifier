"""Combine one run's per-chunk manifests into the run's manifest.csv, the table the model code reads.

The merged manifest has one row per volume, deduplicated by volume_id (an archive
uploaded twice would otherwise appear twice). It carries the volume's kernel and kernel
class (``sharp`` / ``soft`` / ``other``, from the reviewed kernel table), optional
``label_*`` columns (see ingest/labels.py), and the frozen patient-level split. Splits
already frozen by assign_splits.py are joined back in, so re-merging after a top-up never
loses them; patients with no frozen assignment yet are "unassigned".
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from ..config import DataConfig
from ..kernels import add_kernel_columns, load_kernel_table
from ..preprocess import config_fingerprint
from ..runs import Run
from . import state
from .labels import LabelReport, attach_labels
from .splits import apply_splits, read_splits


@dataclass
class MergeReport:
    rows: int = 0
    duplicates_dropped: int = 0
    failures: list[dict] = field(default_factory=list)  # volumes that could not be processed
    kernel_classes: dict[str, int] = field(default_factory=dict)
    labels: LabelReport | None = None


def read_chunk_manifests(run: Run) -> pd.DataFrame:
    folder = run.chunk_manifest_dir
    files = sorted(folder.glob("chunk_*.csv")) if folder.is_dir() else []
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, dtype={"ingest_chunk": str}) for f in files], ignore_index=True)


def collect_failures(run: Run) -> list[dict]:
    out = []
    for chunk_id, info in state.done_chunks(run).items():
        for volume_id, error in (info.get("failed") or {}).items():
            out.append({"source_name": run.source, "ingest_chunk": chunk_id, "volume_id": volume_id, "error": error})
    return out


def _with_kernel_columns(df: pd.DataFrame, table: dict) -> pd.DataFrame:
    """``manufacturer``, ``kernel`` and ``kernel_class`` for either source: CT-RATE's metadata names them
    Manufacturer / ConvolutionKernel, DICOM rows have manufacturer / kernel already."""
    if "Manufacturer" in df.columns and "manufacturer" not in df.columns:
        df = df.rename(columns={"Manufacturer": "manufacturer"})
    kernel_col = "ConvolutionKernel" if "ConvolutionKernel" in df.columns else "kernel"
    if "manufacturer" not in df.columns or kernel_col not in df.columns:
        out = df.copy()
        out["kernel"], out["kernel_class"] = "", "other"
        return out
    return add_kernel_columns(df, table, manufacturer_col="manufacturer", kernel_col=kernel_col)


def merge_run(
    cfg: DataConfig,
    run: Run,
    *,
    kernel_table: dict | None = None,
    labels: pd.DataFrame | None = None,
    manifest_key: str = "volume_id",
    labels_key: str = "volume_id",
) -> tuple[pd.DataFrame, MergeReport]:
    report = MergeReport()
    df = read_chunk_manifests(run)
    if df.empty:
        raise ValueError(f"no chunk manifests found for run {run.name!r} -- run ingest.py --run {run.name} first")
    before = len(df)
    df = df.drop_duplicates("volume_id", keep="first")
    report.duplicates_dropped = before - len(df)
    report.failures = collect_failures(run)

    table = load_kernel_table(cfg.paths.kernel_table) if kernel_table is None else kernel_table
    df = _with_kernel_columns(df, table)
    report.kernel_classes = df["kernel_class"].value_counts().to_dict()
    if labels is not None:
        df, report.labels = attach_labels(df, labels, manifest_key=manifest_key, labels_key=labels_key)

    frozen = {run.source: read_splits(state.splits_path(cfg.paths, run.source))}
    merged = apply_splits(df, frozen)
    report.rows = len(merged)
    return merged, report


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


def write_run_record(run: Run, cfg: DataConfig, manifest: pd.DataFrame, report: MergeReport) -> None:
    """The settings that produced the cache, plus what failed -- a reproducibility
    record (and a fingerprint the model code can compare to know its cache is current)."""
    record = {
        "run": run.name,
        "source": run.source,
        "preprocess_config": asdict(cfg.preprocess),
        "fingerprint": config_fingerprint(cfg.preprocess),
        "n_volumes": int(len(manifest)),
        "kernel_classes": report.kernel_classes,
        "n_failed_volumes": len(report.failures),
        "failed": report.failures,
    }
    state.write_atomic(run.record_path, json.dumps(record, indent=2, default=str))
