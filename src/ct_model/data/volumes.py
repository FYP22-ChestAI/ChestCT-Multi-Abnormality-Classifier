"""Which volumes the model code uses: rows of a run's manifest, resolved against the shared HU cache.

The manifest (docs/preprocessing/data_contract.md) is the only index. Rules applied here:

* only ``split`` in train / val / test (never ``excluded`` / ``unassigned``), and by default only
  ``qc_passed`` rows;
* the cache file is ``<cache_dir>/{volume_id}.npy`` -- the manifest's ``npy_path`` column is NOT used,
  because it is written with the separators of the machine that ran the merge (``data\\cache\\...`` on
  Windows) and would not resolve on the Linux server;
* ``label_<name>`` columns become one float vector per volume, NaN where a label is missing;
* several runs of one source can be used together by joining their names with ``+``
  (``train-sharp-5800+train-soft-1000``): their manifests are concatenated, a volume listed by several
  runs (the shared test pool) is kept once, and a patient must have the same split in every run --
  otherwise it is an error, never a silent leak.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ct_preprocessing.config import DataConfig
from ct_preprocessing.runs import Run, resolve_run

LABEL_PREFIX = "label_"
USABLE_SPLITS = ("train", "val", "test")
RUN_JOIN = "+"  # "runA+runB" = both runs' manifests together


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


@dataclass(frozen=True)
class CombinedRun:
    """Several runs of one source used as one (``data.run: a+b``). Duck-types the fields of a Run that the
    model code reads: ``source``, ``name`` (the joined names) and ``manifest_path`` (for records only)."""
    source: str
    name: str
    parts: tuple[Run, ...]

    @property
    def manifest_path(self) -> str:
        return " + ".join(str(r.manifest_path) for r in self.parts)


def _read_manifest(resolved: Run) -> pd.DataFrame:
    if not resolved.manifest_path.is_file():
        raise FileNotFoundError(
            f"run {resolved.name!r} has no manifest yet ({resolved.manifest_path}) -- "
            "run scripts/preprocessing/merge_manifests.py and assign_splits.py first"
        )
    return pd.read_csv(resolved.manifest_path, low_memory=False)


def combine_manifests(manifests: list[pd.DataFrame], names: list[str]) -> pd.DataFrame:
    """Concatenate run manifests: same label columns required, each volume kept once, and every patient
    in one split across all runs (train / val / test; excluded / unassigned rows are not checked)."""
    labels = [sorted(c for c in m.columns if c.startswith(LABEL_PREFIX)) for m in manifests]
    for name, cols in zip(names[1:], labels[1:]):
        if cols != labels[0]:
            raise ValueError(f"run {name!r} has other label columns than {names[0]!r} -- cannot combine them")
    frame = pd.concat(manifests, ignore_index=True).drop_duplicates("volume_id", keep="first")
    usable = frame[frame["split"].isin(USABLE_SPLITS)]
    splits = usable.groupby("patient_id")["split"].nunique()
    clash = splits[splits > 1].index.tolist()
    if clash:
        raise ValueError(f"{len(clash)} patient(s) have different splits in runs {names} (e.g. {clash[:3]}) -- "
                         "combining them would leak; build the second run with make_worklist.py --from-run")
    return frame.reset_index(drop=True)


def load_run_manifest(data_cfg: DataConfig, source: str, run: str | None = None) -> tuple[Run | CombinedRun, pd.DataFrame]:
    if run and RUN_JOIN in run:
        names = [n.strip() for n in run.split(RUN_JOIN) if n.strip()]
        parts = tuple(resolve_run(data_cfg.paths, source, n) for n in names)
        manifest = combine_manifests([_read_manifest(r) for r in parts], names)
        return CombinedRun(source, RUN_JOIN.join(names), parts), manifest
    resolved = resolve_run(data_cfg.paths, source, run)
    return resolved, _read_manifest(resolved)


def manifest_sha256(run: Run | CombinedRun, file_hash=None) -> str:
    """The manifest's sha256; for a combined run, a sha256 over its parts' hashes (in run order).
    ``file_hash`` (path -> hex) lets a caller pass a cached file hash."""
    if file_hash is None:
        def file_hash(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            return h.hexdigest()
    if isinstance(run, CombinedRun):
        joined = "\n".join(f"{r.name}:{file_hash(r.manifest_path)}" for r in run.parts)
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return file_hash(run.manifest_path)


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
