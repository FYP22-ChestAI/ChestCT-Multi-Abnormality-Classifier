"""DICOM support: read one series into a ``Volume``, and discover scans under
any root folder.

Discovery groups files into scans by their own ``SeriesInstanceUID`` tag, not
by folder location -- matching how SimpleITK's ``GetGDCMSeriesIDs`` and
dcm2niix actually do this (confirmed against their documented behaviour, see
docs/preprocessing/data_contract.md). Concretely: for every folder encountered while
walking the root, if it directly holds DICOM files, those files are grouped
by their series tag *within that folder*. A folder holding exactly one
series is one scan, as before. A folder holding several series (a common,
messy real-world case -- e.g. a single PACS export placing a scout image and
several reconstructions together) now correctly yields one scan per series,
instead of refusing the whole folder. Pointing the root directly at a series
folder (no wrapping subfolder) now works too, for the same reason the old
folder-based approach didn't: nothing depends on folder depth any more.

Nothing here knows or cares which drive, mount or folder layout the files
came from. A Google Drive mount, an external SSD and a university server
folder all work the same way. The only layout-dependent setting is how to
group scans into patients (``patient_key``), and that is a config value, not
code.

DICOM keeps its metadata in tags inside every file (there is no separate
metadata CSV like CT-RATE ships), so everything the pipeline needs --
rescale values, spacing, slice positions -- is read from the files
themselves. HU are always computed explicitly from each slice's own
RescaleSlope/RescaleIntercept: unlike NIfTI, DICOM has no automatic scaling.

Choosing a chest series among several real series is deliberately NOT done
here (the current data has lung series only) -- see docs/preprocessing/data_contract.md
for where that would plug in.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom

from .loader import volume_from_nifti_image
from .types import Volume

DICOM_SUFFIXES = {".dcm", ".dicom", ".ima"}
NIFTI_SUFFIXES = (".nii.gz", ".nii")
_CODEC_HINT = "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"


class DicomReadError(Exception):
    """A DICOM series could not be read safely."""


class MixedSeriesError(DicomReadError):
    """A folder contains more than one DICOM series."""


# --------------------------------------------------------------------------
# file / folder discovery
# --------------------------------------------------------------------------
def is_dicom_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in DICOM_SUFFIXES:
        return True
    if suffix == "":  # extensionless DICOM is common: check the "DICM" marker
        try:
            with open(path, "rb") as f:
                f.seek(128)
                return f.read(4) == b"DICM"
        except OSError:
            return False
    return False


def list_dicom_files(folder: str | Path) -> list[Path]:
    """DICOM files directly inside ``folder`` (not recursive), sorted by name."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.is_file() and is_dicom_file(p))


@dataclass
class ScanEntry:
    scan_path: str  # POSIX path relative to the root, ALWAYS a real folder (or "." for the root itself)
    format: str  # "dicom" (a folder of slices) or "nifti" (one file)
    n_files: int = 1
    # For DICOM: this entry's series tag. Always set when readable. Used to
    # pick out just this series' files if scan_path's folder holds more than
    # one -- never encoded into scan_path itself, so scan_path always stays a
    # literal, directly-openable location.
    series_uid: str | None = None


def _series_uid_of(path: Path) -> str | None:
    """Cheap, header-only read of one file's SeriesInstanceUID. None if the
    file can't be read or has no such tag -- such files are skipped by
    discovery rather than guessed about."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    except Exception:
        return None
    uid = getattr(ds, "SeriesInstanceUID", None)
    return str(uid) if uid else None


def discover_scans(
    root: str | Path,
    fmt: str = "auto",
    only_folders: list[str] | None = None,
    max_scans: int | None = None,
) -> list[ScanEntry]:
    """Find every scan under ``root``, whatever the folder layout.

    fmt: "dicom", "nifti", or "auto" (find both).
    only_folders: keep only scans whose relative path equals, or sits under,
        one of these (e.g. ["4203-26", "4214-26"] or ["4214-26/P00001"]).
    max_scans: keep the first N (after sorting) -- handy for a pilot run.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist or is not a folder: {root}")

    entries: list[ScanEntry] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        here = Path(dirpath)
        rel = here.relative_to(root).as_posix()
        rel_dir = "." if rel == "." else rel  # the root itself is now a valid scan location

        if fmt in ("dicom", "auto"):
            dicom_names = [f for f in filenames if is_dicom_file(here / f)]
            groups: dict[str, list[str]] = {}
            for f in dicom_names:
                uid = _series_uid_of(here / f)
                if uid is None:  # unreadable/corrupt -- skip it, don't let it masquerade as a series
                    continue
                groups.setdefault(uid, []).append(f)
            for uid, files in groups.items():
                entries.append(ScanEntry(scan_path=rel_dir, format="dicom", n_files=len(files), series_uid=uid))
        if fmt in ("nifti", "auto"):
            for f in sorted(filenames):
                if f.lower().endswith(NIFTI_SUFFIXES):
                    r = f"{rel_dir}/{f}" if rel_dir != "." else f
                    entries.append(ScanEntry(scan_path=r, format="nifti"))

    entries.sort(key=lambda e: (e.scan_path, e.series_uid or ""))

    if only_folders:
        wanted = [f.strip().strip("/") for f in only_folders if f.strip()]
        entries = [e for e in entries if any(e.scan_path == w or e.scan_path.startswith(w + "/") for w in wanted)]
    if max_scans:
        entries = entries[: int(max_scans)]
    return entries


def _sanitize(part: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", part)


def make_scan_id(scan_path: str, series_uid: str | None = None) -> str:
    """A filesystem-safe id from a relative scan path: '4214-26/P00001/S0001'
    -> '4214-26_P00001_S0001'. ``series_uid`` is appended (as a short suffix)
    only when given -- needed to keep ids unique when one folder holds
    several series and discover_scans returned more than one entry for it."""
    path = scan_path
    for suffix in NIFTI_SUFFIXES:
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)]
            break
    parts = [p for p in path.split("/") if p and p != "."]
    base = "_".join(_sanitize(p) for p in parts) or "root"
    if series_uid:
        base = f"{base}_s{_sanitize(series_uid)[-8:]}"
    return base


def patient_key(scan_path: str, depth: int = 1) -> str:
    """Group scans into patients by the first ``depth`` folder levels.

    With the NHRD layout '4214-26/P00001/S0001', depth=1 gives '4214-26'
    (one top folder = one patient), whereas 'P00001' alone would collide
    across top folders. A scan sitting directly in the root has no folder
    levels, so it becomes its own patient.
    """
    dir_parts = [p for p in scan_path.split("/") if p][:-1]
    if not dir_parts:
        return make_scan_id(scan_path)
    return "_".join(_sanitize(p) for p in dir_parts[: max(1, depth)])


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------
def _str(ds, name: str) -> str:
    value = getattr(ds, name, "")
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "MultiValue":
        return "|".join(str(v) for v in value)
    return str(value)


def _transfer_syntax(ds) -> str:
    ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if ts is None:
        return "unknown"
    return getattr(ts, "name", str(ts))


def _read_headers(folder: Path, series_uid: str | None = None) -> list[tuple[Path, "pydicom.Dataset"]]:
    """Header-only read of every DICOM file in ``folder``. If ``series_uid``
    is given, only files matching that series are kept -- this is how a
    folder holding several series gets narrowed down to just the one a
    given ScanEntry represents, without ever needing a fake sub-path."""
    files = list_dicom_files(folder)
    if not files:
        raise DicomReadError(f"no DICOM files found directly inside {folder}")
    headers = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception as exc:  # noqa: BLE001 - name the offending file
            raise DicomReadError(f"could not read DICOM header of {p.name}: {exc}") from exc
        if series_uid is not None and _str(ds, "SeriesInstanceUID") != series_uid:
            continue
        headers.append((p, ds))
    if series_uid is not None and not headers:
        raise DicomReadError(f"no files matching series {series_uid!r} found in {folder}")
    return headers


def _slice_geometry(headers) -> dict:
    """Slice order, spacing and orientation, from the tags alone."""
    series_uids = {_str(ds, "SeriesInstanceUID") for _, ds in headers}
    if len(series_uids) > 1:
        raise MixedSeriesError(
            f"folder contains {len(series_uids)} different DICOM series -- one series per folder is "
            "expected (series selection is not implemented; see docs/preprocessing/data_contract.md)"
        )

    first = headers[0][1]
    shapes = {(int(getattr(ds, "Rows", 0)), int(getattr(ds, "Columns", 0))) for _, ds in headers}
    if len(shapes) != 1:
        raise DicomReadError(f"slices in one series have different Rows/Columns: {sorted(shapes)}")

    for name in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing"):
        if any(not hasattr(ds, name) for _, ds in headers):
            raise DicomReadError(f"missing required DICOM tag {name} in at least one slice")

    iop = np.array([float(v) for v in first.ImageOrientationPatient])
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    normal = normal / np.linalg.norm(normal)

    positions = np.array([[float(v) for v in ds.ImagePositionPatient] for _, ds in headers])
    proj = positions @ normal
    order = np.argsort(proj)
    sorted_proj = proj[order]
    diffs = np.diff(sorted_proj)
    if len(diffs) and np.any(diffs < 1e-3):
        raise DicomReadError("duplicate slice positions in this series")

    if len(diffs):
        z_sp = float(np.median(diffs))
        z_uniform = bool(np.ptp(diffs) < 0.1 * z_sp)
        step = (positions[order[-1]] - positions[order[0]]) / (len(order) - 1)
        cos = abs(float(step @ normal)) / float(np.linalg.norm(step))
        tilt = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))
    else:
        z_sp = float(getattr(first, "SliceThickness", 1.0) or 1.0)
        z_uniform, tilt = True, 0.0

    ps = [float(v) for v in first.PixelSpacing]
    return {
        "order": order,
        "positions": positions,
        "row_dir": row_dir,
        "col_dir": col_dir,
        "normal": normal,
        "z_spacing": z_sp,
        "z_uniform": z_uniform,
        "tilt_deg": tilt,
        "row_spacing": ps[0],  # distance between adjacent rows
        "col_spacing": ps[1],  # distance between adjacent columns
        "rows": next(iter(shapes))[0],
        "cols": next(iter(shapes))[1],
    }


def _acquisition_meta(first, n_files: int) -> dict:
    """Non-identifying acquisition fields worth keeping (proposal 4.4: keep
    acquisition metadata rather than normalising it away). Deliberately a
    small whitelist -- no patient, date, institution or physician tags."""
    return {
        "n_files": n_files,
        "manufacturer": _str(first, "Manufacturer"),
        "model": _str(first, "ManufacturerModelName"),
        "kernel": _str(first, "ConvolutionKernel"),
        "slice_thickness": _str(first, "SliceThickness"),
        "series_description": _str(first, "SeriesDescription"),
        "contrast": bool(_str(first, "ContrastBolusAgent")),
        "transfer_syntax": _transfer_syntax(first),
    }


def summarize_dicom_folder(folder: str | Path, series_uid: str | None = None) -> dict:
    """Header-only summary of one scan (no pixel data is decoded, so this is
    fast). Feeds build_manifest_folder()'s acquisition columns AND its
    automatic patient_id_source="auto" decision, from the same reads.
    ``n_series`` always reports how many DISTINCT series exist in the whole
    folder, even when ``series_uid`` narrows the rest of the summary to just
    one of them -- so a mixed folder is still visibly flagged as such."""
    folder = Path(folder)
    all_headers = _read_headers(folder)
    out_n_series = len({_str(ds, "SeriesInstanceUID") for _, ds in all_headers})
    headers = _read_headers(folder, series_uid=series_uid) if series_uid else all_headers
    first = headers[0][1]
    out = _acquisition_meta(first, len(headers))
    out["n_series"] = out_n_series
    out["rows"], out["cols"] = int(getattr(first, "Rows", 0)), int(getattr(first, "Columns", 0))
    out["photometric"] = _str(first, "PhotometricInterpretation")
    out["modality"] = _str(first, "Modality")
    try:
        g = _slice_geometry(headers)
        out.update(z_spacing=g["z_spacing"], z_uniform=g["z_uniform"], tilt_deg=g["tilt_deg"],
                   row_spacing=g["row_spacing"], col_spacing=g["col_spacing"])
    except DicomReadError as exc:
        out["geometry_problem"] = str(exc)
    pid = _str(first, "PatientID")
    out["patient_id_present"] = bool(pid)
    # a short hash, so uniqueness across folders can be judged without printing the value
    out["patient_id_hash"] = hashlib.sha256(pid.encode()).hexdigest()[:12] if pid else ""
    out["has_study_uid"] = bool(_str(first, "StudyInstanceUID"))
    out["has_series_uid"] = bool(_str(first, "SeriesInstanceUID"))
    return out


def load_dicom_series(folder: str | Path, volume_id: str | None = None, series_uid: str | None = None) -> Volume:
    """Read one DICOM series into a (Z, Y, X) HU ``Volume`` in RAS+.

    ``folder`` is the directory the files live in; ``series_uid`` picks out
    just one series if that folder holds more than one (from a ScanEntry
    returned by discover_scans). If not given, the folder must hold exactly
    one series, or MixedSeriesError is raised -- unchanged behaviour for the
    common, tidy one-series-per-folder case.

    Slices are ordered by their real position along the scan axis (not by
    file name), each slice's own RescaleSlope/RescaleIntercept is applied,
    and the geometry is handed to the SAME nibabel-based standardisation the
    NIfTI loader uses, so both formats end up in an identical convention.
    """
    folder = Path(folder)
    headers = _read_headers(folder, series_uid=series_uid)
    g = _slice_geometry(headers)
    first = headers[0][1]

    if _str(first, "PhotometricInterpretation") not in ("MONOCHROME2", ""):
        raise DicomReadError(
            f"PhotometricInterpretation {_str(first, 'PhotometricInterpretation')!r} is not supported "
            "(only MONOCHROME2 CT data is)"
        )
    if int(getattr(first, "SamplesPerPixel", 1)) != 1:
        raise DicomReadError("multi-sample (colour) DICOM is not supported for CT")

    rows, cols, n = g["rows"], g["cols"], len(headers)
    vol = np.empty((n, rows, cols), dtype=np.float32)
    rescale_pairs = set()
    for k, idx in enumerate(g["order"]):
        path = headers[idx][0]
        ds = pydicom.dcmread(str(path))
        try:
            arr = ds.pixel_array
        except Exception as exc:  # noqa: BLE001 - usually a missing decoder for a compressed syntax
            raise DicomReadError(
                f"cannot decode pixel data of {path.name} (transfer syntax: {_transfer_syntax(ds)}): {exc}. "
                f"If the files are compressed, install decoders: {_CODEC_HINT}"
            ) from exc
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        rescale_pairs.add((slope, intercept))
        vol[k] = arr.astype(np.float32) * slope + intercept

    # Geometry: voxel (i=column, j=row, k=slice) -> patient position, in LPS
    # (DICOM's convention), then to RAS (nibabel's) by flipping x and y.
    x_step = g["row_dir"] * g["col_spacing"]  # moving along a row = next column
    y_step = g["col_dir"] * g["row_spacing"]  # moving down a column = next row
    z_step = g["normal"] * g["z_spacing"]
    origin = g["positions"][g["order"][0]]
    affine_lps = np.eye(4)
    affine_lps[:3, 0], affine_lps[:3, 1], affine_lps[:3, 2], affine_lps[:3, 3] = x_step, y_step, z_step, origin
    affine_ras = np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine_lps

    data_ijk = np.transpose(vol, (2, 1, 0))  # (cols, rows, slices)
    img = nib.Nifti1Image(data_ijk, affine_ras)

    meta = {
        "source_path": str(folder),
        "volume_id": volume_id or make_scan_id(folder.name, series_uid),
        "format": "dicom",
        "series_uid": _str(first, "SeriesInstanceUID"),
        **_acquisition_meta(first, n),
        "z_uniform": g["z_uniform"],
        "z_spacing_median": g["z_spacing"],
        "slice_axis_tilt_deg": g["tilt_deg"],
        "rescale_varies": len(rescale_pairs) > 1,
    }
    return volume_from_nifti_image(img, meta=meta)
