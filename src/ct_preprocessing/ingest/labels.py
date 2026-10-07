"""Abnormality labels, joined into a run's manifest at merge time.

The labels are known up front and are small, so putting them in the manifest gives later
stages one table with split, kernel and labels side by side. They are ONLY joined here:
preprocessing, worklists and splits never read them, so nothing about which scans are kept
or which patient goes where can depend on a label.

  * CT-RATE: two small CSVs (train / valid, one row per volume, 18 abnormality columns) are
    downloaded next to the metadata. They are *predicted* labels (parsed from the radiology
    reports by a model), not human-checked ground truth.
  * archive sources (NHRD): a labels CSV sits in the same Drive folder as the archives. Its
    first column identifies the scan (``scan_path``, ``patient_id`` or ``volume_id``, see
    ``LabelsConfig``); every other column becomes a label.

Every label column of the manifest is called ``label_<name>`` (lower case, underscores).
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import LabelsConfig, PathsConfig
from ..manifest import strip_nifti_ext
from .base import FetchError

CTRATE_LABEL_FILES = {
    "train": "dataset/multi_abnormality_labels/train_predicted_labels.csv",
    "valid": "dataset/multi_abnormality_labels/valid_predicted_labels.csv",
}


def label_column(name: str) -> str:
    """``"Medical material"`` -> ``"label_medical_material"``."""
    return "label_" + re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


@dataclass
class LabelReport:
    columns: list[str] = field(default_factory=list)
    rows: int = 0
    rows_with_labels: int = 0
    unmatched_label_keys: int = 0
    duplicate_label_keys: int = 0

    def format(self) -> str:
        text = f"labels: {len(self.columns)} column(s), {self.rows_with_labels} of {self.rows} manifest rows labelled"
        if self.unmatched_label_keys:
            text += f", {self.unmatched_label_keys} label row(s) match no manifest row"
        if self.duplicate_label_keys:
            text += f", {self.duplicate_label_keys} duplicate label key(s) (first kept)"
        return text


# ---------------------------------------------------------------------------- CT-RATE
def ctrate_label_paths(paths: PathsConfig) -> dict[str, Path]:
    return {pool: Path(paths.metadata_dir) / Path(repo_file).name for pool, repo_file in CTRATE_LABEL_FILES.items()}


def download_ctrate_labels(
    paths: PathsConfig, repo_id: str, hf_hub_download: Callable, *, refresh: bool = False
) -> list[str]:
    """Fetch the two label CSVs (~3 MB) unless already present. Labels are optional, so a file that
    cannot be fetched is reported (the returned warnings) and never stops the worklist; the CSVs can
    also be copied by hand into the metadata folder."""
    warnings = []
    for pool, dest in ctrate_label_paths(paths).items():
        if dest.exists() and not refresh:
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            cached = hf_hub_download(repo_id=repo_id, filename=CTRATE_LABEL_FILES[pool], repo_type="dataset")
            shutil.copyfile(cached, dest)
        except Exception as exc:  # noqa: BLE001 - network, login or a path that moved: labels are optional
            warnings.append(
                f"could not download {CTRATE_LABEL_FILES[pool]} ({type(exc).__name__}: {str(exc)[:120]}); "
                f"copy {dest.name} into {dest.parent} by hand to get labels in the manifest"
            )
    return warnings


def load_ctrate_labels(paths: PathsConfig) -> pd.DataFrame | None:
    """Both pools' labels as one frame keyed by ``volume_id``; None if no label file is present."""
    frames = []
    for dest in ctrate_label_paths(paths).values():
        if dest.exists():
            df = pd.read_csv(dest)
            df = df.rename(columns={"VolumeName": "volume_id"})
            df["volume_id"] = df["volume_id"].map(strip_nifti_ext)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


# ------------------------------------------------------------------- archive sources
def fetch_archive_labels(backend, labels_cfg: LabelsConfig, dest: str | Path) -> Path:
    """Copy the labels CSV out of the archives' folder (Drive via rclone, or a plain folder)."""
    if not labels_cfg.file:
        raise ValueError("this source has no labels file configured (sources.<name>.labels.file)")
    dest = Path(dest)
    try:
        backend.get(labels_cfg.file, dest)
    except FetchError:
        raise
    except OSError as exc:
        raise FetchError(f"cannot read {labels_cfg.file}: {exc}") from exc
    return dest


def read_labels_csv(path: str | Path) -> pd.DataFrame:
    """A labels CSV: first column the key, the rest labels."""
    df = pd.read_csv(path, dtype={0: str})
    if df.shape[1] < 2:
        raise ValueError(f"{path}: expected a key column followed by at least one label column")
    return df


# ------------------------------------------------------------------------------- join
def attach_labels(manifest: pd.DataFrame, labels: pd.DataFrame, *, manifest_key: str, labels_key: str) -> tuple[pd.DataFrame, LabelReport]:
    """The manifest with ``label_*`` columns, matched on ``manifest[manifest_key] == labels[labels_key]``."""
    if manifest_key not in manifest.columns:
        raise ValueError(f"the manifest has no {manifest_key!r} column to match labels on")
    if labels_key not in labels.columns:
        raise ValueError(f"the labels file has no {labels_key!r} column (it has: {', '.join(map(str, labels.columns[:6]))}...)")
    labels = labels.copy()
    labels[labels_key] = labels[labels_key].astype(str).str.strip().str.rstrip("/\\")
    duplicates = int(labels[labels_key].duplicated().sum())
    labels = labels.drop_duplicates(labels_key, keep="first")
    renamed = {c: label_column(c) for c in labels.columns if c != labels_key}
    labels = labels.rename(columns=renamed)
    value_columns = list(renamed.values())

    keys = manifest[manifest_key].astype(str).str.strip().str.rstrip("/\\")
    out = manifest.drop(columns=[c for c in value_columns if c in manifest.columns]).copy()
    lookup = labels.set_index(labels_key)[value_columns]
    for column in value_columns:
        out[column] = keys.map(lookup[column])
    matched = keys.isin(lookup.index)
    return out, LabelReport(
        columns=value_columns,
        rows=len(out),
        rows_with_labels=int(matched.sum()),
        unmatched_label_keys=int((~labels[labels_key].isin(set(keys))).sum()),
        duplicate_label_keys=duplicates,
    )
