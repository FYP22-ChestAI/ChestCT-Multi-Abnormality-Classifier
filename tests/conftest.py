"""Shared test fixtures: a crude synthetic "chest" volume, so tests don't
need a real CT-RATE download to check the pipeline logic."""
from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest


def make_synthetic_chest_hu(shape=(40, 64, 64)) -> np.ndarray:
    """An elliptical soft-tissue "body" (~40 HU) in an air background
    (-1000 HU), with two lower-density "lung" regions (~-800 HU) inside it,
    away from the outer edge. Every z-slice has the same cross-section.

    Built specifically so tests can check that cropping keeps the WHOLE body
    even though the lungs read as background-like air in the middle.
    """
    z, y, x = shape
    _, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    cy, cx = y / 2, x / 2

    body = ((yy - cy) ** 2 / (y * 0.35) ** 2 + (xx - cx) ** 2 / (x * 0.35) ** 2) < 1.0

    hu = np.full(shape, -1000.0, dtype=np.float32)
    hu[body] = 40.0

    for sign in (-1, 1):
        lcy = cy + sign * y * 0.15
        lung = ((yy - lcy) ** 2 / (y * 0.12) ** 2 + (xx - cx) ** 2 / (x * 0.15) ** 2) < 1.0
        hu[lung & body] = -800.0

    return hu


def write_dicom_series(
    folder,
    hu_zyx,
    spacing_zyx=(1.5, 1.0, 1.0),
    iop=(1, 0, 0, 0, 1, 0),
    intercept=-1024.0,
    intercepts=None,
    slope=1.0,
    series_uid=None,
    shuffle_names=True,
    skip_slices=(),
    name_prefix="IMG",
    transfer_syntax=None,
    patient_id="ANON01",
    seed=0,
):
    """Write a (Z, Y, X) HU array as a real DICOM series (one file per slice).

    ``iop`` is the ImageOrientationPatient (row direction, then column
    direction, in DICOM's LPS coordinates). File names are shuffled by
    default so they do NOT match slice order -- the loader must sort by
    position, not by name.
    """
    import numpy as np
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    folder.mkdir(parents=True, exist_ok=True)
    series_uid = series_uid or generate_uid()
    study_uid = generate_uid()
    n, rows, cols = hu_zyx.shape
    row_dir, col_dir = np.array(iop[:3], float), np.array(iop[3:], float)
    normal = np.cross(row_dir, col_dir)

    names = list(range(n))
    if shuffle_names:
        np.random.default_rng(seed).shuffle(names)

    for k in range(n):
        if k in skip_slices:
            continue
        this_intercept = float(intercepts[k]) if intercepts is not None else intercept
        raw = np.rint((hu_zyx[k] - this_intercept) / slope)
        assert raw.min() >= 0 and raw.max() <= 65535, "test volume does not fit uint16 with this intercept"

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = transfer_syntax or ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.PatientID = patient_id
        ds.Modality = "CT"
        ds.Manufacturer = "TESTCO"
        ds.ManufacturerModelName = "SYNTH-1"
        ds.ConvolutionKernel = "B70f"
        ds.SeriesDescription = "Lung synthetic"
        ds.SliceThickness = spacing_zyx[0]
        ds.InstanceNumber = k + 1
        ds.Rows, ds.Columns = rows, cols
        ds.PixelSpacing = [spacing_zyx[1], spacing_zyx[2]]  # [row spacing, column spacing]
        ds.ImageOrientationPatient = list(iop)
        ds.ImagePositionPatient = list(normal * (k * spacing_zyx[0]))
        ds.RescaleSlope = slope
        ds.RescaleIntercept = this_intercept
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelData = raw.astype(np.uint16).tobytes()
        ds.save_as(str(folder / f"{name_prefix}-0001-{names[k] + 1:05d}.dcm"))
    return series_uid


@pytest.fixture
def dicom_writer():
    return write_dicom_series


@pytest.fixture
def synthetic_hu():
    return make_synthetic_chest_hu()


@pytest.fixture
def synthetic_nifti(tmp_path):
    """Write make_synthetic_chest_hu() out as a real .nii.gz, spacing (1.5, 1.0, 1.0)
    mm in (z, y, x), so loader/preprocess tests exercise real file I/O."""
    hu = make_synthetic_chest_hu()
    spacing_zyx = (1.5, 1.0, 1.0)

    # nibabel expects data as (i, j, k) with affine mapping (i,j,k) -> (x,y,z).
    # Build a plain positive-diagonal affine so as_closest_canonical is a no-op
    # and our (Z, Y, X) array round-trips predictably through the loader.
    affine = np.diag([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0], 1.0])
    data_ijk = np.transpose(hu, (2, 1, 0)).astype(np.float32)

    path = tmp_path / "synthetic_1_a_1.nii.gz"
    nib.save(nib.Nifti1Image(data_ijk, affine), str(path))
    return path, hu, spacing_zyx
