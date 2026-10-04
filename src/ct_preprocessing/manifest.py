"""Build manifest rows: one row per volume, no labels, no split.

A manifest row says what a volume IS (its ids, where it came from, how it was
acquired) -- never which split it belongs to. Splits are decided once, later,
by ct_preprocessing.ingest.splits, after ingest and QC, and frozen.

No series selection happens here -- CT-RATE files are already the chest. What
we do handle: several *reconstructions* of the same scan (they are not
different body parts, just different rebuilds of the same raw data -- see
docs/preprocessing/data_contract.md), and a ``patient_id`` on every row so the
later split can be made by PATIENT and no patient's scans land in two splits.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .dicom_loader import DicomReadError, ScanEntry, make_scan_id, patient_key, summarize_dicom_folder

CTRATE_BUILDER = "ctrate"
FOLDER_BUILDER = "folder"

# CT-RATE volume ids look like "train_1_a_1": pool, patient, scan, reconstruction.
VOLUME_ID_RE = re.compile(r"^(?P<split>train|valid)_(?P<patient>\d+)_(?P<scan>[a-z]+)_(?P<recon>\d+)$")

ACQUISITION_COLUMNS = [
    "rows", "cols", "manufacturer", "model", "kernel",
    "slice_thickness", "series_description", "contrast", "transfer_syntax",
]


def parse_volume_id(volume_id: str) -> dict:
    """Split 'train_1_a_2' into its parts.

    patient_id is pool-qualified ("train_1", not just "1") so a patient
    number is never accidentally treated as the same person across CT-RATE's
    official train/valid pools. ``source_split`` records which official pool
    the volume came from ("train" or "valid").
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


def strip_nifti_ext(value: object) -> str:
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
    """CT-RATE rows: parsed ids (train_1_a_1 style) + optional acquisition
    metadata (RescaleSlope/Intercept, spacing, ...) joined on VolumeName. No
    labels -- this pipeline never touches them, for either data source; the
    model code joins labels on its own side, by volume_id, when it needs them
    for supervision. See docs/preprocessing/data_contract.md.
    """
    manifest = pd.DataFrame([parse_volume_id(v) for v in volume_ids])

    if metadata_df is not None:
        metadata_df = metadata_df.copy()
        if id_col in metadata_df.columns:
            metadata_df[id_col] = metadata_df[id_col].map(strip_nifti_ext)
            metadata_df = metadata_df.set_index(id_col)
        manifest = manifest.join(metadata_df, on="volume_id", rsuffix="_meta")

    # Where each scan lives, relative to the source's scratch folder (never absolute).
    manifest["scan_path"] = manifest["volume_id"] + ".nii.gz"
    manifest["format"] = "nifti"
    return manifest


def build_manifest_folder(
    root: str | Path,
    entries: list[ScanEntry],
    patient_depth: int = 1,
    patient_id_source: str = "auto",
) -> pd.DataFrame:
    """One row per discovered scan, for any folder of NIfTI files or DICOM
    series (e.g. the local NHRD data). No labels and no split.

    Identity comes from the relative scan_path plus, for DICOM, a short suffix
    of that series' own SeriesInstanceUID (never a fake sub-path -- scan_path
    always stays a literal, directly-openable folder), so a folder holding
    several series yields several unique rows.

    Patients are grouped by folder depth or by the DICOM PatientID tag.
    ``patient_id_source="auto"`` decides between them itself, from the SAME
    per-scan tag reads this function already does for the acquisition columns
    (no extra file reads): it only switches to tag-based grouping when
    PatientID is present AND distinct across every DICOM scan found. Real
    hospital data is routinely anonymised before being shared for research,
    which strips this tag entirely -- defaulting to the always-safe
    folder-depth fallback and only upgrading on positive, confirming evidence
    avoids silently grouping by a tag that turns out to be missing or (worse)
    accidentally shared between two different real patients. Pass "path" or
    "dicom_tag" to force one without the auto-check. The choice actually used
    is recorded in ``manifest.attrs["patient_id_source"]`` so a caller that
    builds a source in several pieces can keep it consistent across them.

    For DICOM the acquisition fields are read from the tags (there is no
    metadata CSV); dates and identifying tags are deliberately not recorded.
    """
    root = Path(root)
    rows: list[dict] = []
    dicom_rows: list[tuple[dict, dict]] = []  # (row, summary) -- reused for the auto-check below

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
                if "geometry_problem" in summary:
                    row["manifest_problem"] = summary["geometry_problem"]
                dicom_rows.append((row, summary))
            except DicomReadError as exc:
                row["manifest_problem"] = str(exc)
        rows.append(row)

    resolved_source = patient_id_source
    if patient_id_source == "auto":
        hashes = [s.get("patient_id_hash", "") for _, s in dicom_rows]
        present = sum(1 for h in hashes if h)
        distinct = len({h for h in hashes if h})
        if dicom_rows and present == len(hashes) and distinct == present:
            resolved_source = "dicom_tag"
            print(f"PatientID present and distinct on {present}/{len(hashes)} DICOM scan(s) -- using patient_id_source=dicom_tag")
        else:
            resolved_source = "path"
            print(
                f"PatientID missing or not distinct on {present}/{len(hashes)} DICOM scan(s) -- "
                f"using patient_id_source=path (patient_path_depth={patient_depth})"
            )

    if resolved_source == "dicom_tag":
        for row, summary in dicom_rows:
            if summary.get("patient_id_present"):
                row["patient_id"] = "dicom_" + summary["patient_id_hash"]

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError(f"no scans to put in the manifest (root: {root})")
    dup = manifest[manifest["volume_id"].duplicated(keep=False)]
    if len(dup):
        raise ValueError(f"scan ids are not unique: {dup['scan_path'].tolist()[:6]}")
    manifest.attrs["patient_id_source"] = resolved_source
    return manifest


def check_no_patient_overlap(manifest: pd.DataFrame, split_col: str = "split", patient_col: str = "patient_id") -> None:
    """Raise if any patient's scans appear in more than one split."""
    by_patient = manifest.groupby(patient_col)[split_col].nunique()
    bad = by_patient[by_patient > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} patients appear in more than one split: {bad.index.tolist()[:5]}")
