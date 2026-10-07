"""Which volumes the model code uses: rows of a run's manifest, resolved against the shared HU cache.

The manifest (docs/preprocessing/data_contract.md) is the only index. Rules applied here:

* only ``split`` in train / val / test (never ``excluded`` / ``unassigned``), and by default only
  ``qc_passed`` rows;
* the cache file is ``<cache_dir>/{volume_id}.npy`` -- the manifest's ``npy_path`` column is NOT used,
  because it is written with the separators of the machine that ran the merge (``data\\cache\\...`` on
  Windows) and would not resolve on the Linux server;
* ``label_<name>`` columns become one float vector per volume, NaN where a label is missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ct_preprocessing.config import DataConfig
from ct_preprocessing.runs import Run, resolve_run

LABEL_PREFIX = "label_"
USABLE_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class VolumeRecord:
    volume_id: str
    patient_id: str
    split: str
    kernel_class: str
    n_slices: int
    npy_path: Path
    labels: tuple[float, ...] = ()  # in the order of label_names(manifest); NaN = missing


def label_names(manifest: pd.DataFrame) -> list[str]:
    """The label names (without the ``label_`` prefix), in manifest column order."""
    return [c[len(LABEL_PREFIX):] for c in manifest.columns if c.startswith(LABEL_PREFIX)]


def load_run_manifest(data_cfg: DataConfig, source: str, run: str | None = None) -> tuple[Run, pd.DataFrame]:
    resolved = resolve_run(data_cfg.paths, source, run)
    if not resolved.manifest_path.is_file():
        raise FileNotFoundError(
            f"run {resolved.name!r} has no manifest yet ({resolved.manifest_path}) -- "
            "run scripts/preprocessing/merge_manifests.py and assign_splits.py first"
        )
    return resolved, pd.read_csv(resolved.manifest_path, low_memory=False)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def select_volumes(
    manifest: pd.DataFrame,
    cache_dir: str | Path,
    *,
    splits: tuple[str, ...] = USABLE_SPLITS,
    qc_passed_only: bool = True,
    kernel_classes: tuple[str, ...] | None = None,
) -> list[VolumeRecord]:
    """The manifest rows to use, as records, in manifest order (deduplicated by volume_id)."""
    bad = set(splits) - set(USABLE_SPLITS)
    if bad:
        raise ValueError(f"splits must be among {list(USABLE_SPLITS)}, got {sorted(bad)}")
    missing_cols = {"volume_id", "patient_id", "split", "n_slices"} - set(manifest.columns)
    if missing_cols:
        raise ValueError(f"manifest lacks columns {sorted(missing_cols)}")

    rows = manifest[manifest["split"].isin(splits)]
    if qc_passed_only and "qc_passed" in rows.columns:
        rows = rows[_as_bool(rows["qc_passed"])]
    if kernel_classes is not None:
        if "kernel_class" not in rows.columns:
            raise ValueError("kernel_classes was given but the manifest has no kernel_class column")
        rows = rows[rows["kernel_class"].isin(kernel_classes)]
    rows = rows.drop_duplicates("volume_id")

    label_cols = [c for c in manifest.columns if c.startswith(LABEL_PREFIX)]
    labels = rows[label_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32) if label_cols else None
    kernel = rows["kernel_class"] if "kernel_class" in rows.columns else pd.Series("", index=rows.index)
    cache_dir = Path(cache_dir)
    return [
        VolumeRecord(
            volume_id=str(vid),
            patient_id=str(pid),
            split=str(split),
            kernel_class="" if pd.isna(kc) else str(kc),
            n_slices=int(n),
            npy_path=cache_dir / f"{vid}.npy",
            labels=tuple(float(v) for v in labels[i]) if labels is not None else (),
        )
        for i, (vid, pid, split, kc, n) in enumerate(
            zip(rows["volume_id"], rows["patient_id"], rows["split"], kernel, rows["n_slices"])
        )
    ]


def missing_cache_files(records: list[VolumeRecord]) -> list[str]:
    """volume_ids whose cache .npy is not on disk (the manifest promises it, so this is an error)."""
    return [r.volume_id for r in records if not r.npy_path.is_file()]
