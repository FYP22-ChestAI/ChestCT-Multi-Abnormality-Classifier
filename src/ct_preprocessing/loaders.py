"""Choose the loader by file format. This is the only place that knows there
are two formats; everything after it works on the shared ``Volume``.

A "scan" is a single ``.nii``/``.nii.gz`` file (NIfTI) or one DICOM series
(a folder, optionally narrowed to one ``series_uid`` if that folder holds
more than one -- see dicom_loader.py). If the format isn't given it is
detected from the path.
"""
from __future__ import annotations

from pathlib import Path

from .dicom_loader import NIFTI_SUFFIXES, _series_uid_of, list_dicom_files, load_dicom_series
from .loader import load_nifti_volume
from .types import Volume


def detect_format(scan_path: str | Path) -> str:
    p = Path(scan_path)
    if p.is_dir():
        return "dicom"
    if p.name.lower().endswith(NIFTI_SUFFIXES):
        return "nifti"
    raise ValueError(f"cannot tell the scan format of {p} (expected a NIfTI file or a DICOM folder)")


def load_volume(
    scan_path: str | Path, fmt: str | None = None, volume_id: str | None = None, series_uid: str | None = None
) -> Volume:
    fmt = fmt or detect_format(scan_path)
    if fmt == "nifti":
        return load_nifti_volume(scan_path, volume_id=volume_id)
    if fmt == "dicom":
        return load_dicom_series(scan_path, volume_id=volume_id, series_uid=series_uid)
    raise ValueError(f"unknown scan format {fmt!r} (expected 'nifti' or 'dicom')")


def source_signature(scan_path: str | Path, fmt: str | None = None, series_uid: str | None = None) -> str:
    """A cheap fingerprint of the raw input: file count and total bytes.

    Recorded next to each cached result so a later run can tell that the RAW
    data changed (e.g. a re-download from a different CT-RATE folder under
    the same volume ids) even when the preprocessing config did not. Size
    rather than modification time on purpose: copying a folder from Drive to
    local disk changes every mtime but not the content.

    ``series_uid`` narrows this to just that series' files when the folder
    holds more than one -- otherwise a change to an unrelated series in the
    same folder would look like this scan changed too.
    """
    p = Path(scan_path)
    fmt = fmt or detect_format(p)
    if fmt == "dicom":
        files = list_dicom_files(p)
        if series_uid:
            files = [f for f in files if _series_uid_of(f) == series_uid]
        return f"{len(files)}:{sum(f.stat().st_size for f in files)}"
    return f"1:{p.stat().st_size}"
