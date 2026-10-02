"""The CT-RATE worklist: exactly which volumes to fetch, grouped into chunks.

This is a work list, NOT a split. Which patients end up in train, val or test
is decided later, once, by ct_preprocessing.ingest.splits, after ingest and QC.
All the worklist decides is *what is worth ingesting* when the whole dataset
will not fit on disk:

  * the test pool (CT-RATE's official valid_fixed/ pool) is taken whole by
    default, so results stay comparable with published CT-RATE numbers (it can
    be capped for a small pilot run);
  * the train pool is taken one reconstruction per scan by default (the two
    reconstructions of a scan are near-duplicate rebuilds of the same raw
    data), in a seeded random order of PATIENTS, optionally capped.

The order is a seeded patient shuffle and a cap simply truncates it, so the
worklist is *prefix-stable*: raising or lowering ``max_train_patients`` (or
``max_test_patients``) later never changes a chunk that was already ingested.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd

from ..manifest import parse_volume_id, strip_nifti_ext
from .ctrate import ctrate_rel_path, passes_memory_screen

WORKLIST_COLUMNS = [
    "chunk", "order", "volume_id", "patient_id", "scan_id", "reconstruction_id", "source_split", "repo_path",
]


class WorklistConflict(RuntimeError):
    """Rewriting the worklist would change chunks that were already ingested."""


def _catalog(meta: pd.DataFrame, pool: str) -> tuple[pd.DataFrame, dict]:
    """Parse one pool's metadata into one row per volume, dropping rows that do
    not belong (unparsable ids, or an id from the other pool)."""
    rows, unparsable, wrong_pool = [], 0, 0
    for name in meta["VolumeName"].map(strip_nifti_ext):
        try:
            parsed = parse_volume_id(name)
        except ValueError:
            unparsable += 1
            continue
        if parsed["source_split"] != pool:
            wrong_pool += 1
            continue
        rows.append(parsed)
    df = pd.DataFrame(rows, columns=["volume_id", "patient_id", "scan_id", "reconstruction_id", "source_split"])
    return df, {"unparsable": unparsable, "wrong_pool": wrong_pool}


def _natural_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_pnum"] = df["patient_id"].str.extract(r"_(\d+)$")[0].astype(int)
    return df.sort_values(["_pnum", "scan_id", "reconstruction_id"]).drop(columns="_pnum")


def _one_per_scan(df: pd.DataFrame) -> pd.DataFrame:
    return _natural_sort(df).drop_duplicates(["patient_id", "scan_id"], keep="first")


def _seeded_patient_order(df: pd.DataFrame, seed: int, cap: int | None) -> pd.DataFrame:
    """Rows ordered by a seeded shuffle of PATIENTS (a patient's volumes stay
    together), truncated to the first ``cap`` patients. A cap only truncates
    that order, which is what makes the worklist prefix-stable."""
    patients = sorted(df["patient_id"].unique())
    random.Random(seed).shuffle(patients)
    if cap is not None:
        patients = patients[:cap]
    rank = {p: i for i, p in enumerate(patients)}
    out = df[df["patient_id"].isin(rank)].copy()
    out["_rank"] = out["patient_id"].map(rank)
    return out.sort_values(["_rank", "scan_id", "reconstruction_id"]).drop(columns="_rank")


def build_worklist(
    train_meta: pd.DataFrame,
    valid_meta: pd.DataFrame,
    *,
    chunk_size: int,
    train_pool: str = "one_per_scan",
    test_pool: str = "all",
    max_train_patients: int | None = None,
    max_test_patients: int | None = None,
    max_combined_gb: float | None = None,
    target_spacing_zyx: tuple[float, float, float] = (1.5, 0.75, 0.75),
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Returns (worklist, summary). Test-pool chunks come first, then train;
    a chunk never mixes the two pools."""
    train, train_skipped = _catalog(train_meta, "train")
    test, test_skipped = _catalog(valid_meta, "valid")
    summary = {
        "skipped_unparsable": train_skipped["unparsable"] + test_skipped["unparsable"],
        "skipped_wrong_pool": train_skipped["wrong_pool"] + test_skipped["wrong_pool"],
        "train_volumes_available": len(train),
        "train_patients_available": train["patient_id"].nunique(),
        "screened_out": 0,
    }

    if max_combined_gb:  # TRAIN only: silently dropping test volumes would bias the benchmark
        meta_indexed = train_meta.assign(VolumeName=train_meta["VolumeName"].map(strip_nifti_ext)).set_index("VolumeName")
        keep = [passes_memory_screen(v, meta_indexed, max_combined_gb, target_spacing_zyx) for v in train["volume_id"]]
        summary["screened_out"] = len(train) - sum(keep)
        train = train[keep]

    if train_pool == "one_per_scan":
        train = _one_per_scan(train)
    if test_pool == "one_per_scan":
        test = _one_per_scan(test)
    summary["test_patients_available"] = test["patient_id"].nunique()

    train = _seeded_patient_order(train, seed, max_train_patients)
    test = _seeded_patient_order(test, seed + 1, max_test_patients)

    n_test_chunks = math.ceil(len(test) / chunk_size)
    test = test.reset_index(drop=True)
    train = train.reset_index(drop=True)
    test["chunk"] = test.index // chunk_size
    train["chunk"] = n_test_chunks + train.index // chunk_size

    worklist = pd.concat([test, train], ignore_index=True)
    worklist["order"] = worklist.index
    worklist["repo_path"] = [ctrate_rel_path(v, p) for v, p in zip(worklist["volume_id"], worklist["source_split"])]
    worklist = worklist[WORKLIST_COLUMNS]

    summary.update(
        test_volumes=len(test), test_patients=test["patient_id"].nunique(), test_chunks=n_test_chunks,
        train_volumes=len(train), train_patients=train["patient_id"].nunique(),
        train_chunks=worklist["chunk"].nunique() - n_test_chunks,
        total_volumes=len(worklist), total_chunks=worklist["chunk"].nunique(),
    )
    return worklist, summary


def format_summary(summary: dict, est_mb_per_volume: float) -> str:
    gb = summary["total_volumes"] * est_mb_per_volume / 1000
    lines = [
        f"test  (valid pool): {summary['test_volumes']:>6} volumes, {summary['test_patients']:>5} patients, {summary['test_chunks']} chunks"
        f"  (of {summary['test_patients_available']} patients available)",
        f"train (train pool): {summary['train_volumes']:>6} volumes, {summary['train_patients']:>5} patients, {summary['train_chunks']} chunks"
        f"  (of {summary['train_patients_available']} patients available)",
        f"total: {summary['total_volumes']} volumes in {summary['total_chunks']} chunks -- projected cache ~{gb:.0f} GB "
        f"at {est_mb_per_volume:g} MB/volume (measure the real value with a one-chunk calibration run)",
    ]
    if summary["screened_out"]:
        lines.append(f"{summary['screened_out']} train volume(s) skipped by the memory screen")
    if summary["skipped_unparsable"] or summary["skipped_wrong_pool"]:
        lines.append(
            f"ignored metadata rows: {summary['skipped_unparsable']} unparsable ids, "
            f"{summary['skipped_wrong_pool']} ids from the wrong pool"
        )
    return "\n".join(lines)


def read_worklist(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run scripts/preprocessing/make_worklist.py first")
    return pd.read_csv(path)


def write_worklist(df: pd.DataFrame, path: str | Path, *, overwrite: bool = False, done_chunks=()) -> None:
    """Write the worklist. An existing one is only replaced with ``overwrite``,
    and then only if every already-ingested chunk keeps exactly the same
    volumes -- otherwise a "done" marker would silently cover different scans."""
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --force to rebuild it")
        old = pd.read_csv(path)
        label = lambda c: f"{int(c):04d}"  # noqa: E731
        old_by_chunk = old.groupby(old["chunk"].map(label))["volume_id"].apply(list)
        new_by_chunk = df.groupby(df["chunk"].map(label))["volume_id"].apply(list)
        changed = [c for c in done_chunks if c in old_by_chunk.index and old_by_chunk[c] != new_by_chunk.get(c)]
        if changed:
            raise WorklistConflict(
                f"rebuilding would change chunk(s) already ingested: {sorted(changed)[:8]}. Their volumes would no "
                "longer match their done markers. Delete those markers in the ingest state folder to re-ingest them, "
                "or keep the settings that affect chunk contents (chunk_size, pools, seed) unchanged."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
