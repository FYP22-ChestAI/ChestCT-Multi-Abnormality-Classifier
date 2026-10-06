"""The CT-RATE worklist: exactly which volumes a run fetches, grouped into chunks.

This is a work list, NOT a split. Which patients end up in train, val or test
is decided later, once, by ct_preprocessing.ingest.splits, after ingest and QC.
All the worklist decides is *what is worth ingesting* when the whole dataset
will not fit on disk:

  * the test pool (CT-RATE's official valid_fixed/ pool) is always taken whole, with
    EVERY reconstruction (both kernels), so results stay comparable with published
    CT-RATE numbers and can be reported per kernel (it can be capped for a small pilot);
  * the train pool is taken one reconstruction per scan, of the kernel the run asks for
    (``sharp`` or ``soft``); a scan without that kernel is skipped. Patients come in a
    seeded random order and a cap truncates it.

Because the order is a seeded shuffle of ALL train patients (not only those that have the
wanted kernel), a sharp run and a soft run with the same seed walk the same patient order,
and a cap only truncates it -- raising or lowering ``max_train_patients`` later never
reorders what was already chosen.

A run can also be built from a FINISHED run (``select_child_patients``): a subset of the
patients the parent already split, with a chosen number from its train split and from its
val split. Every patient keeps the split it already has, so the new run cannot leak.
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import pandas as pd

from ..kernels import SHARP, SOFT, add_kernel_columns
from ..manifest import parse_volume_id, strip_nifti_ext
from .ctrate import ctrate_rel_path, passes_memory_screen

WORKLIST_COLUMNS = [
    "chunk", "order", "volume_id", "patient_id", "scan_id", "reconstruction_id", "source_split",
    "manufacturer", "kernel", "kernel_class", "repo_path",
]
_CATALOG_COLUMNS = ["volume_id", "patient_id", "scan_id", "reconstruction_id", "source_split"]
_NEEDED_METADATA = ("VolumeName", "Manufacturer", "ConvolutionKernel")


def _catalog(meta: pd.DataFrame, pool: str, table: dict) -> tuple[pd.DataFrame, dict]:
    """Parse one pool's metadata into one row per volume (with its manufacturer, kernel and kernel
    class), dropping rows that do not belong (unparsable ids, or an id from the other pool)."""
    missing = [c for c in _NEEDED_METADATA if c not in meta.columns]
    if missing:
        raise ValueError(f"the {pool} metadata CSV has no column(s) {missing}")
    rows, unparsable, wrong_pool = [], 0, 0
    for name, manufacturer, kernel in zip(meta["VolumeName"].map(strip_nifti_ext), meta["Manufacturer"], meta["ConvolutionKernel"]):
        try:
            parsed = parse_volume_id(name)
        except ValueError:
            unparsable += 1
            continue
        if parsed["source_split"] != pool:
            wrong_pool += 1
            continue
        rows.append({**parsed, "manufacturer": manufacturer, "_raw_kernel": kernel})
    df = pd.DataFrame(rows, columns=[*_CATALOG_COLUMNS, "manufacturer", "_raw_kernel"])
    df = add_kernel_columns(df, table, manufacturer_col="manufacturer", kernel_col="_raw_kernel").drop(columns="_raw_kernel")
    df = df.drop_duplicates("volume_id")
    return df, {"unparsable": unparsable, "wrong_pool": wrong_pool}


def _natural_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_pnum"] = df["patient_id"].str.extract(r"_(\d+)$")[0].astype(int)
    return df.sort_values(["_pnum", "scan_id", "reconstruction_id"]).drop(columns="_pnum")


def _one_per_scan(df: pd.DataFrame) -> pd.DataFrame:
    return _natural_sort(df).drop_duplicates(["patient_id", "scan_id"], keep="first")


def _shuffled(patients, seed: int) -> list[str]:
    ordered = sorted(set(patients))
    random.Random(seed).shuffle(ordered)
    return ordered


def _rows_in_patient_order(df: pd.DataFrame, ordered_patients: list[str]) -> pd.DataFrame:
    """``df``'s rows grouped by patient in the given order (a patient's volumes stay together)."""
    rank = {p: i for i, p in enumerate(ordered_patients)}
    out = df[df["patient_id"].isin(rank)].copy()
    out["_rank"] = out["patient_id"].map(rank)
    return out.sort_values(["_rank", "scan_id", "reconstruction_id"]).drop(columns="_rank")


def select_child_patients(
    parent_splits: dict[str, str], available: set[str], *, n_train: int, n_val: int, seed: int
) -> list[str]:
    """Patients for a run built from a finished run: ``n_train`` drawn from the parent's train split and
    ``n_val`` from its val split (seeded), restricted to ``available`` (patients that have the wanted
    kernel). Every chosen patient keeps the split the parent gave it, so training on them cannot leak
    into the validation or test patients, and the train / val mix is what you asked for, not luck."""
    pools = {
        s: sorted(p for p, split in parent_splits.items() if split == s and p in available) for s in ("train", "val")
    }
    for s, n in (("train", n_train), ("val", n_val)):
        if n > len(pools[s]):
            raise ValueError(
                f"asked for {n} {s} patients but only {len(pools[s])} patients of the parent's {s} split "
                "have the wanted kernel and are in the parent run"
            )
    rng = random.Random(seed)
    chosen = []
    for s, n in (("train", n_train), ("val", n_val)):
        pool = list(pools[s])
        rng.shuffle(pool)
        chosen += pool[:n]
    rng.shuffle(chosen)  # mix the two splits through the chunks
    return chosen


def patients_with_kernel(train_meta: pd.DataFrame, kernel_table: dict, kernel: str) -> set[str]:
    """Train-pool patients that have at least one scan with a ``kernel`` reconstruction."""
    train, _ = _catalog(train_meta, "train", kernel_table)
    return set(train.loc[train["kernel_class"] == kernel, "patient_id"])


def build_worklist(
    train_meta: pd.DataFrame,
    valid_meta: pd.DataFrame,
    *,
    kernel_table: dict,
    chunk_size: int,
    train_kernel: str = SHARP,
    max_train_patients: int | None = None,
    max_test_patients: int | None = None,
    max_combined_gb: float | None = None,
    target_spacing_zyx: tuple[float, float, float] = (1.5, 0.75, 0.75),
    seed: int = 0,
    train_patients: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Returns (worklist, summary). Test-pool chunks come first, then train; a chunk never mixes
    the two pools. ``train_patients`` (from select_child_patients) fixes the train patients and their
    order instead of the seeded shuffle."""
    if train_kernel not in (SHARP, SOFT):
        raise ValueError(f"train_kernel must be 'sharp' or 'soft', got {train_kernel!r}")
    train, train_skipped = _catalog(train_meta, "train", kernel_table)
    test, test_skipped = _catalog(valid_meta, "valid", kernel_table)
    summary = {
        "train_kernel": train_kernel,
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

    all_train_patients = train["patient_id"].unique()
    summary["train_scans_total"] = len(train.drop_duplicates(["patient_id", "scan_id"]))
    wanted = _one_per_scan(train[train["kernel_class"] == train_kernel])
    summary["train_scans_selected"] = len(wanted)
    summary["train_scans_skipped_no_kernel"] = summary["train_scans_total"] - len(wanted)
    have_kernel = set(wanted["patient_id"])
    summary["train_patients_without_kernel"] = len(set(all_train_patients) - have_kernel)

    if train_patients is not None:
        ordered = [p for p in train_patients if p in have_kernel]
    else:
        ordered = [p for p in _shuffled(all_train_patients, seed) if p in have_kernel]
        if max_train_patients is not None:
            ordered = ordered[:max_train_patients]
    train_rows = _rows_in_patient_order(wanted, ordered)

    test_order = _shuffled(test["patient_id"].unique(), seed + 1)
    if max_test_patients is not None:
        test_order = test_order[:max_test_patients]
    summary["test_patients_available"] = test["patient_id"].nunique()
    test_rows = _rows_in_patient_order(test, test_order)  # every reconstruction, whatever its kernel

    n_test_chunks = math.ceil(len(test_rows) / chunk_size)
    test_rows = test_rows.reset_index(drop=True)
    train_rows = train_rows.reset_index(drop=True)
    test_rows["chunk"] = test_rows.index // chunk_size
    train_rows["chunk"] = n_test_chunks + train_rows.index // chunk_size

    worklist = pd.concat([test_rows, train_rows], ignore_index=True)
    worklist["order"] = worklist.index
    worklist["repo_path"] = [ctrate_rel_path(v, p) for v, p in zip(worklist["volume_id"], worklist["source_split"])]
    worklist = worklist[WORKLIST_COLUMNS]

    summary.update(
        test_volumes=len(test_rows), test_patients=test_rows["patient_id"].nunique(), test_chunks=n_test_chunks,
        test_kernel_classes=test_rows["kernel_class"].value_counts().to_dict(),
        train_volumes=len(train_rows), train_patients=train_rows["patient_id"].nunique(),
        train_chunks=worklist["chunk"].nunique() - n_test_chunks,
        total_volumes=len(worklist), total_chunks=worklist["chunk"].nunique(),
    )
    return worklist, summary


def build_survey_worklist(
    train_meta: pd.DataFrame, *, kernel_table: dict, pairs_per_kernel: int, chunk_size: int, seed: int = 0
) -> tuple[pd.DataFrame, dict]:
    """A small worklist for the kernel survey: for every (manufacturer, kernel) pair, up to
    ``pairs_per_kernel`` scans that have two reconstructions, with BOTH reconstructions of each.
    After ingest, ``kernel_survey.py`` measures which reconstruction of each pair is the sharper one,
    which checks the sharp / soft table against data instead of against anyone's memory."""
    train, _ = _catalog(train_meta, "train", kernel_table)
    sizes = train.groupby(["patient_id", "scan_id"]).size()
    paired = set(sizes[sizes == 2].index)
    scans = train[[(p, s) in paired for p, s in zip(train["patient_id"], train["scan_id"])]]
    rng = random.Random(seed)
    # a scan covers every (manufacturer, kernel) pair among its two volumes, so one scan can serve two pairs
    covers: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for p, s, m, k in zip(scans["patient_id"], scans["scan_id"], scans["manufacturer"], scans["kernel"]):
        covers.setdefault((p, s), set()).add((m, k))
    by_pair: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for scan, pairs in covers.items():
        for pair in pairs:
            by_pair.setdefault(pair, []).append(scan)
    chosen: set[tuple[str, str]] = set()
    for pair in sorted(by_pair, key=lambda q: (len(by_pair[q]), q)):  # rarest pairs first, so they are not crowded out
        candidates = sorted(by_pair[pair])
        rng.shuffle(candidates)
        have = sum(1 for scan in chosen if pair in covers[scan])
        for scan in candidates:
            if have >= pairs_per_kernel:
                break
            if scan not in chosen:
                chosen.add(scan)
                have += 1
    pairs_covered = len(by_pair)
    rows = scans[[(p, s) in chosen for p, s in zip(scans["patient_id"], scans["scan_id"])]]
    rows = _natural_sort(rows).reset_index(drop=True)
    rows["chunk"] = rows.index // chunk_size
    rows["order"] = rows.index
    rows["repo_path"] = [ctrate_rel_path(v, p) for v, p in zip(rows["volume_id"], rows["source_split"])]
    summary = {
        "survey_scans": len(chosen), "survey_volumes": len(rows), "kernel_pairs": pairs_covered,
        "total_volumes": len(rows), "total_chunks": int(rows["chunk"].nunique()) if len(rows) else 0,
    }
    return rows[WORKLIST_COLUMNS], summary


def format_summary(summary: dict, est_mb_per_volume: float) -> str:
    gb = summary["total_volumes"] * est_mb_per_volume / 1000
    if "survey_scans" in summary:
        return (
            f"kernel survey: {summary['survey_scans']} scans ({summary['survey_volumes']} volumes, both reconstructions "
            f"of each) covering {summary['kernel_pairs']} (manufacturer, kernel) pairs, {summary['total_chunks']} chunks "
            f"-- cache ~{gb:.1f} GB at {est_mb_per_volume:g} MB/volume"
        )
    classes = ", ".join(f"{k}={v}" for k, v in sorted(summary["test_kernel_classes"].items()))
    lines = [
        f"test  (valid pool, every reconstruction): {summary['test_volumes']:>6} volumes ({classes}), "
        f"{summary['test_patients']:>5} patients, {summary['test_chunks']} chunks"
        f"  (of {summary['test_patients_available']} patients available)",
        f"train (train pool, {summary['train_kernel']} only): {summary['train_volumes']:>6} volumes, {summary['train_patients']:>5} patients, "
        f"{summary['train_chunks']} chunks  (of {summary['train_patients_available']} patients available)",
        f"  {summary['train_scans_selected']} of {summary['train_scans_total']} train scans have a {summary['train_kernel']} reconstruction; "
        f"{summary['train_scans_skipped_no_kernel']} scans skipped ({summary['train_patients_without_kernel']} patients have none)",
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


def write_worklist(df: pd.DataFrame, path: str | Path) -> None:
    """Write a run's worklist. A worklist is never replaced: a new plan is a new run."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists -- worklists are never overwritten; make a new run instead")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
