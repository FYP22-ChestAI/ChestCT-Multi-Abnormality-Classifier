"""Train / val / test split: by PATIENT, decided once after ingest + QC, then frozen.

Why last, and why this is leak-free: nothing in this pipeline is fitted to the
data (fixed spacing, fixed crop threshold, fixed size -- no dataset-wide mean
or std), so preprocessing before the split cannot leak anything. Splitting
afterwards also means QC-failed scans are excluded *before* patients are
assigned, so val/test counts mean "usable patients".

The assignment is written to ``<splits_dir>/<source>.csv`` and never rewritten:
re-running only assigns patients that are not in the file yet (a later
top-up of the dataset), leaving every earlier assignment untouched. Freeze it
before the first training run and never change it afterwards.

CT-RATE's test set is its own official valid pool -- a rule, not a draw. Only
its validation set is drawn (from train-pool patients). A source with no
official split (NHRD) draws both val and test.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import SourceConfig
from ..manifest import CTRATE_BUILDER, check_no_patient_overlap
from .state import write_atomic

SPLITS = ("train", "val", "test")


class InsufficientPatients(ValueError):
    """Fewer unassigned patients are available than the requested val/test counts need."""


def plan_split(
    patients: list[str],
    *,
    n_val: int,
    n_test: int,
    seed: int,
    existing: dict[str, str] | None = None,
    forced: dict[str, str] | None = None,
) -> dict[str, str]:
    """The full patient -> split mapping, keeping every ``existing`` assignment.

    ``forced`` assignments are rules (CT-RATE's valid pool -> test), applied
    to new patients and not counted against the n_val/n_test quotas. The
    quotas are TOTALS: a later top-up only fills whatever is still missing,
    and every other new patient goes to train.
    """
    existing = dict(existing or {})
    forced = forced or {}
    bad = {s for s in existing.values() if s not in SPLITS}
    if bad:
        raise ValueError(f"existing split file has unknown split label(s): {sorted(bad)}")

    result = dict(existing)
    new = sorted(set(patients) - set(existing))
    for p in new:
        if p in forced:
            result[p] = forced[p]
    candidates = [p for p in new if p not in forced]

    have_val = sum(1 for p, s in existing.items() if s == "val" and p not in forced)
    have_test = sum(1 for p, s in existing.items() if s == "test" and p not in forced)
    need_val, need_test = max(0, n_val - have_val), max(0, n_test - have_test)
    if need_val + need_test > len(candidates):
        raise InsufficientPatients(
            f"need {need_test} test + {need_val} val patients but only {len(candidates)} unassigned patient(s) "
            f"are available (already assigned: {have_test} test, {have_val} val)"
        )
    random.Random(seed).shuffle(candidates)
    for p in candidates[:need_test]:
        result[p] = "test"
    for p in candidates[need_test : need_test + need_val]:
        result[p] = "val"
    for p in candidates[need_test + need_val :]:
        result[p] = "train"
    return result


def read_splits(path: str | Path) -> dict[str, str]:
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    return dict(zip(df["patient_id"], df["split"]))


def write_splits(path: str | Path, mapping: dict[str, str]) -> None:
    df = pd.DataFrame(sorted(mapping.items()), columns=["patient_id", "split"])
    write_atomic(Path(path), df.to_csv(index=False))


@dataclass
class SplitReport:
    source: str
    mapping: dict[str, str]
    n_new_patients: int
    n_kept_patients: int
    n_unusable_volumes: int
    patients_per_split: dict[str, int] = field(default_factory=dict)
    volumes_per_split: dict[str, int] = field(default_factory=dict)

    def format(self) -> str:
        rows = [
            f"{s:<6} {self.patients_per_split.get(s, 0):>6} patients {self.volumes_per_split.get(s, 0):>7} volumes"
            for s in SPLITS
        ]
        head = (
            f"[{self.source}] {self.n_new_patients} patient(s) newly assigned, {self.n_kept_patients} already frozen; "
            f"{self.n_unusable_volumes} volume(s) excluded (failed QC / not preprocessed)"
        )
        return "\n".join([head, *rows])


def assign_source_splits(
    manifest: pd.DataFrame,
    source_name: str,
    source_cfg: SourceConfig,
    splits_file: str | Path,
    qc_report: pd.DataFrame | None,
    *,
    n_val: int | None = None,
    n_test: int | None = None,
    seed: int | None = None,
    dry_run: bool = False,
) -> SplitReport:
    """Assign (or top up) the frozen split for one source, from the merged manifest.

    ``qc_report`` (the qc_report.csv frame) decides which volumes are usable:
    only QC-passed volumes count, so a patient whose scans all failed is not
    assigned at all. Pass None to skip that filter (the caller must say so).
    """
    rows = manifest[manifest["source_name"] == source_name]
    if rows.empty:
        raise ValueError(f"no manifest rows for source {source_name!r} -- run merge_manifests.py first")
    if qc_report is not None:
        passed = set(qc_report.loc[qc_report["passed"].astype(bool), "volume_id"])
        usable = rows[rows["volume_id"].isin(passed)]
    else:
        usable = rows

    n_val = source_cfg.split.n_val_patients if n_val is None else n_val
    n_test = (source_cfg.split.n_test_patients or 0) if n_test is None else n_test
    if n_val is None:
        raise ValueError(f"no val patient count: set sources.{source_name}.split.n_val_patients or pass --n-val-patients")
    seed = source_cfg.split.seed if seed is None else seed

    patients = sorted(usable["patient_id"].unique())
    forced: dict[str, str] = {}
    if source_cfg.manifest_builder == CTRATE_BUILDER:
        official_test = usable.loc[usable["source_split"] == "valid", "patient_id"].unique()
        forced = {p: "test" for p in official_test}

    existing = read_splits(splits_file)
    mapping = plan_split(patients, n_val=n_val, n_test=n_test, seed=seed, existing=existing, forced=forced)
    if not dry_run:
        write_splits(splits_file, mapping)

    vol_split = usable["patient_id"].map(mapping)
    return SplitReport(
        source=source_name,
        mapping=mapping,
        n_new_patients=len(set(mapping) - set(existing)),
        n_kept_patients=len(existing),
        n_unusable_volumes=len(rows) - len(usable),
        patients_per_split={s: sum(1 for p in patients if mapping.get(p) == s) for s in SPLITS},
        volumes_per_split={s: int((vol_split == s).sum()) for s in SPLITS},
    )


def apply_splits(
    manifest: pd.DataFrame,
    splits_by_source: dict[str, dict[str, str]],
    qc_report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The manifest with ``split`` (train/val/test, "excluded" for a volume that
    failed QC, "unassigned" for a patient with no frozen assignment yet) and,
    when a QC report is given, ``qc_passed`` -- then checked for patient overlap."""
    out = manifest.drop(columns=[c for c in ("split",) if c in manifest.columns]).copy()
    if qc_report is not None:
        passed = {v: bool(p) for v, p in zip(qc_report["volume_id"], qc_report["passed"])}
        out["qc_passed"] = out["volume_id"].map(passed)  # NaN: volume absent from the QC report

    def split_of(row) -> str:
        if qc_report is not None:
            qc = row["qc_passed"]
            if not (pd.notna(qc) and bool(qc)):
                return "excluded"
        return splits_by_source.get(row["source_name"], {}).get(row["patient_id"], "unassigned")

    out["split"] = out.apply(split_of, axis=1)
    assigned = out[out["split"].isin(SPLITS)]
    check_no_patient_overlap(assigned.assign(patient_id=assigned["source_name"] + "/" + assigned["patient_id"]))
    return out
