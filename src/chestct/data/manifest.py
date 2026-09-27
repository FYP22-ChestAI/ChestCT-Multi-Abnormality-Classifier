"""Step 3: build the one-row-per-volume manifest and patient-level splits.

No series selection happens here -- CT-RATE files are already the chest.
What we do handle: several *reconstructions* of the same scan (they are not
different body parts, just different rebuilds of the same raw data -- see
docs/data_contract.md), and splitting by PATIENT so no patient's scans land
in two splits at once.
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

from .dicom_loader import DicomReadError, ScanEntry, make_scan_id, patient_key, summarize_dicom_folder

# CT-RATE volume ids look like "train_1_a_1": split, patient, scan, reconstruction.
VOLUME_ID_RE = re.compile(r"^(?P<split>train|valid)_(?P<patient>\d+)_(?P<scan>[a-z]+)_(?P<recon>\d+)$")


def parse_volume_id(volume_id: str) -> dict:
    """Split 'train_1_a_1' into its parts.

    patient_id is split-qualified ("train_1", not just "1") so a patient
    number is never accidentally treated as the same person across the
    official train/valid halves.
    """
    m = VOLUME_ID_RE.match(volume_id)
    if not m:
        raise ValueError(f"volume id does not match the CT-RATE naming pattern: {volume_id!r}")
    d = m.groupdict()
    return {
        "volume_id": volume_id,
        "patient_id": f"{d['split']}_{d['patient']}",
        "scan_id": d["scan"],
        "reconstruction_id": int(d["recon"]),
        "source_split": d["split"],
    }


def _strip_nifti_ext(value: object) -> str:
    """CT-RATE's own CSVs store the volume id WITH the file extension (e.g.
    'train_1_a_1.nii.gz'), but everywhere else in this package uses the bare
    id ('train_1_a_1'). Normalise both sides before joining so it doesn't
    matter which form a given CSV happens to use."""
    s = str(value)
    for ext in (".nii.gz", ".nii"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def build_manifest_ctrate(
    volume_ids: list[str],
    metadata_df: pd.DataFrame | None = None,
    id_col: str = "VolumeName",
) -> pd.DataFrame:
    """CT-RATE's manifest-building strategy: parsed ids (train_1_a_1 style) +
    optional acquisition metadata (RescaleSlope/Intercept, spacing, ...),
    joined on VolumeName. No labels -- this pipeline never touches them, for
    either data source; M4 joins labels on its own side, by volume_id, when
    it needs them for supervision. See docs/data_contract.md.

    This is one entry in MANIFEST_BUILDERS, not the only possible one -- a
    future local (NHRD) source has a completely different id scheme and
    registers its own strategy here instead of this function growing special
    cases. See configs/data.yaml's `sources` section.
    """
    rows = [parse_volume_id(v) for v in volume_ids]
    manifest = pd.DataFrame(rows)

    if metadata_df is not None and id_col in metadata_df.columns:
        metadata_df = metadata_df.copy()
        metadata_df[id_col] = metadata_df[id_col].map(_strip_nifti_ext)
    if metadata_df is not None:
        meta_indexed = metadata_df.set_index(id_col) if id_col in metadata_df.columns else metadata_df
        manifest = manifest.join(meta_indexed, on="volume_id", rsuffix="_meta")

    # Where each scan lives, relative to the source's root folder (never absolute).
    manifest["scan_path"] = manifest["volume_id"] + ".nii.gz"
    manifest["format"] = "nifti"
    return manifest


# Manifest builders that start from a list of known volume ids plus tables
# (CT-RATE's labels/metadata CSVs). The other kind -- "folder" -- discovers
# scans by walking a root directory instead, so it has its own function
# (build_manifest_folder) rather than sharing this call shape; the script
# scripts/build_manifest.py picks between them from the source's
# `manifest_builder` setting in configs/data.yaml.
MANIFEST_BUILDERS = {"ctrate": build_manifest_ctrate}
FOLDER_BUILDER = "folder"

ACQUISITION_COLUMNS = [
    "rows", "cols", "manufacturer", "model", "kernel",
    "slice_thickness", "series_description", "contrast", "transfer_syntax",
]


def build_manifest_folder(
    root: str | Path,
    entries: list[ScanEntry],
    patient_depth: int = 1,
    patient_id_source: str = "path",
) -> pd.DataFrame:
    """One row per discovered scan, for any folder of NIfTI files or DICOM
    series (e.g. the local NHRD data). No labels, and no split assigned here
    -- both are the caller's job (scripts/build_manifest.py), matching how
    build_manifest_ctrate works, so the two stay symmetric.

    Identity comes from the relative scan_path plus, when a folder holds more
    than one series, a short suffix of that series' own tag (never a fake
    sub-path -- scan_path always stays a literal, directly-openable folder).
    Patients are grouped by folder depth or by the DICOM PatientID tag -- see
    SourceConfig. For DICOM the acquisition fields are read from the tags
    (there is no metadata CSV); date and identifying tags are deliberately
    not recorded.
    """
    root = Path(root)
    rows = []
    for e in entries:
        vid = make_scan_id(e.scan_path, e.series_uid if e.format == "dicom" else None)
        row = {
            "volume_id": vid,
            "patient_id": patient_key(e.scan_path, patient_depth),
            "scan_path": e.scan_path,
            "format": e.format,
            "n_files": e.n_files,
            "series_uid": e.series_uid if e.format == "dicom" else None,
        }
        if e.format == "dicom":
            try:
                summary = summarize_dicom_folder(root / e.scan_path, series_uid=e.series_uid)
                for col in ACQUISITION_COLUMNS:
                    row[col] = summary.get(col)
                if patient_id_source == "dicom_tag" and summary.get("patient_id_present"):
                    row["patient_id"] = "dicom_" + summary["patient_id_hash"]
                if "geometry_problem" in summary:
                    row["manifest_problem"] = summary["geometry_problem"]
            except DicomReadError as exc:
                row["manifest_problem"] = str(exc)
        rows.append(row)

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"no scans to put in the manifest (root: {root})")
    dup = manifest[manifest["volume_id"].duplicated(keep=False)]
    if len(dup):
        raise ValueError(f"scan ids are not unique: {dup['scan_path'].tolist()[:6]}")
    return manifest


def build_manifest(
    volume_ids: list[str],
    metadata_df: pd.DataFrame | None = None,
    id_col: str = "VolumeName",
    builder: str = "ctrate",
) -> pd.DataFrame:
    """Entry point: dispatches to MANIFEST_BUILDERS[builder]. No labels are
    ever read or joined here -- see build_manifest_ctrate's docstring."""
    if builder not in MANIFEST_BUILDERS:
        raise ValueError(f"unknown manifest builder {builder!r}; known: {sorted(MANIFEST_BUILDERS)}")
    return MANIFEST_BUILDERS[builder](volume_ids, metadata_df, id_col)


def assign_patient_splits(patient_ids: list[str], val_fraction: float = 0.1, seed: int = 0) -> dict[str, str]:
    """Split PATIENTS (not volumes) into train/val.

    This only carves a validation set out of CT-RATE's *train* patients.
    CT-RATE's own valid patients should be kept as the held-out benchmark set
    (assigned "test" by the caller), never mixed back into train/val.
    """
    unique = sorted(set(patient_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction))) if unique else 0
    val_patients = set(unique[:n_val])
    return {p: ("val" if p in val_patients else "train") for p in unique}


def assign_splits_by_amount(
    patient_ids: list[str],
    n_train: int,
    n_val: int,
    n_test: int = 0,
    seed: int = 0,
) -> dict[str, str]:
    """Randomly assign exactly ``n_train``/``n_val``/``n_test`` PATIENTS
    (not scans) into their groups, by a fixed seed for reproducibility.

    Used for both CT-RATE (called on the official train-source patients,
    n_test=0, since CT-RATE's official valid-source patients become "test"
    directly, with no further splitting -- see docs/data_contract.md) and
    local data (called once on the whole discovered pool, all three amounts
    together, since local data has no separate official train/valid halves).

    Patients not selected at all are simply left out of the returned dict --
    e.g. when the three amounts add up to less than the full available pool
    (a smaller pilot run). Raises if there aren't enough distinct patients to
    satisfy what was asked for.
    """
    unique = sorted(set(patient_ids))
    total = n_train + n_val + n_test
    if total > len(unique):
        raise ValueError(
            f"asked for {total} patients (train={n_train}, val={n_val}, test={n_test}) "
            f"but only {len(unique)} distinct patients are available"
        )
    rng = random.Random(seed)
    rng.shuffle(unique)
    chosen = unique[:total]
    result: dict[str, str] = {}
    for pid in chosen[:n_train]:
        result[pid] = "train"
    for pid in chosen[n_train : n_train + n_val]:
        result[pid] = "val"
    for pid in chosen[n_train + n_val : total]:
        result[pid] = "test"
    return result


def check_no_patient_overlap(manifest: pd.DataFrame, split_col: str = "split", patient_col: str = "patient_id") -> None:
    """Raise if any patient's scans appear in more than one split."""
    by_patient = manifest.groupby(patient_col)[split_col].nunique()
    bad = by_patient[by_patient > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} patients appear in more than one split: {bad.index.tolist()[:5]}")
