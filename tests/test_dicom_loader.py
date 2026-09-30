"""DICOM loader tests, all on synthetic series written by conftest.write_dicom_series."""
import numpy as np
import pytest

from chestct.preprocessing.dicom_loader import (
    DicomReadError,
    MixedSeriesError,
    discover_scans,
    load_dicom_series,
    make_scan_id,
    patient_key,
    summarize_dicom_folder,
)


def test_dicom_matches_the_nifti_convention_exactly(tmp_path, dicom_writer, synthetic_hu):
    """The NIfTI fixture holds this same volume with x->Right, y->Anterior, z->Superior.
    Writing it as DICOM with the matching orientation must load back identically."""
    folder = tmp_path / "S0001"
    dicom_writer(folder, synthetic_hu, spacing_zyx=(1.5, 1.0, 1.0), iop=(-1, 0, 0, 0, -1, 0))

    vol = load_dicom_series(folder)

    assert vol.orientation == "RAS+"
    assert vol.hu.shape == synthetic_hu.shape
    np.testing.assert_allclose(vol.hu, synthetic_hu)
    assert vol.spacing == pytest.approx((1.5, 1.0, 1.0), abs=1e-4)


def test_standard_dicom_orientation_is_flipped_into_ras(tmp_path, dicom_writer, synthetic_hu):
    # A standard axial DICOM (rows run toward Posterior, columns toward Left) must be
    # flipped in-plane to reach the same RAS+ convention.
    folder = tmp_path / "S0001"
    dicom_writer(folder, synthetic_hu, iop=(1, 0, 0, 0, 1, 0))

    vol = load_dicom_series(folder)

    np.testing.assert_allclose(vol.hu, synthetic_hu[:, ::-1, ::-1])


def test_slices_are_ordered_by_position_not_file_name(tmp_path, dicom_writer, synthetic_hu):
    a, b = tmp_path / "shuffled", tmp_path / "ordered"
    dicom_writer(a, synthetic_hu, shuffle_names=True)
    dicom_writer(b, synthetic_hu, shuffle_names=False)
    np.testing.assert_allclose(load_dicom_series(a).hu, load_dicom_series(b).hu)


def test_each_slices_own_rescale_is_applied(tmp_path, dicom_writer, synthetic_hu):
    intercepts = [-1024.0 if k % 2 == 0 else -2000.0 for k in range(synthetic_hu.shape[0])]
    folder = tmp_path / "S0001"
    dicom_writer(folder, synthetic_hu, intercepts=intercepts, iop=(-1, 0, 0, 0, -1, 0))
    vol = load_dicom_series(folder)
    np.testing.assert_allclose(vol.hu, synthetic_hu)
    assert vol.meta["rescale_varies"] is True


def test_missing_slice_is_reported_as_nonuniform_spacing(tmp_path, dicom_writer, synthetic_hu):
    folder = tmp_path / "S0001"
    dicom_writer(folder, synthetic_hu, skip_slices=(10,))
    vol = load_dicom_series(folder)
    assert vol.meta["z_uniform"] is False


def test_two_series_in_one_folder_are_refused(tmp_path, dicom_writer, synthetic_hu):
    folder = tmp_path / "S0001"
    dicom_writer(folder, synthetic_hu, name_prefix="A")
    dicom_writer(folder, synthetic_hu, name_prefix="B")  # a different SeriesInstanceUID
    with pytest.raises(MixedSeriesError):
        load_dicom_series(folder)


def test_empty_folder_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DicomReadError):
        load_dicom_series(tmp_path / "empty")


def test_undecodable_pixel_data_explains_what_to_install(tmp_path, dicom_writer, synthetic_hu):
    import pydicom
    from pydicom.encaps import encapsulate
    from pydicom.uid import JPEG2000Lossless

    folder = tmp_path / "S0001"
    dicom_writer(folder, synthetic_hu[:3])
    for p in folder.glob("*.dcm"):  # rewrite each slice claiming JPEG 2000 with garbage data
        ds = pydicom.dcmread(str(p))
        ds.file_meta.TransferSyntaxUID = JPEG2000Lossless
        ds.PixelData = encapsulate([b"not really jpeg2000"])
        ds["PixelData"].is_undefined_length = True
        ds.save_as(str(p))
    with pytest.raises(DicomReadError, match="transfer syntax"):
        load_dicom_series(folder)


def test_summary_reads_headers_without_identifying_tags(tmp_path, dicom_writer, synthetic_hu):
    folder = tmp_path / "S0001"
    dicom_writer(folder, synthetic_hu, patient_id="SOME-ID")
    s = summarize_dicom_folder(folder)
    assert s["n_files"] == 40 and s["n_series"] == 1
    assert s["manufacturer"] == "TESTCO" and s["kernel"] == "B70f"
    assert s["patient_id_present"] is True
    assert len(s["patient_id_hash"]) == 12 and "SOME-ID" not in str(s.values())
    assert s["z_uniform"] is True


# ---------------------------------------------------------------- discovery
def _layout(tmp_path, dicom_writer, synthetic_hu):
    """Two 'patients' with the NHRD-style layout, plus noise files."""
    for top in ("4203-26", "4214-26"):
        dicom_writer(tmp_path / top / "P00001" / "S0001", synthetic_hu[:5])
    (tmp_path / "4203-26" / "notes.txt").write_text("not a scan")
    (tmp_path / ".hidden").mkdir()
    return tmp_path


def test_discover_finds_scan_folders_whatever_the_layout(tmp_path, dicom_writer, synthetic_hu):
    root = _layout(tmp_path, dicom_writer, synthetic_hu)
    found = discover_scans(root)
    assert [e.scan_path for e in found] == ["4203-26/P00001/S0001", "4214-26/P00001/S0001"]
    assert all(e.format == "dicom" and e.n_files == 5 for e in found)


def test_discover_works_on_a_different_layout(tmp_path, dicom_writer, synthetic_hu):
    # a shallower, differently named drive -- no code change needed
    dicom_writer(tmp_path / "case_a", synthetic_hu[:5])
    dicom_writer(tmp_path / "batch2" / "x" / "y" / "case_b", synthetic_hu[:5])
    assert [e.scan_path for e in discover_scans(tmp_path)] == ["batch2/x/y/case_b", "case_a"]


def test_discover_only_folders_and_max_scans(tmp_path, dicom_writer, synthetic_hu):
    root = _layout(tmp_path, dicom_writer, synthetic_hu)
    assert [e.scan_path for e in discover_scans(root, only_folders=["4214-26"])] == ["4214-26/P00001/S0001"]
    assert len(discover_scans(root, max_scans=1)) == 1


def test_discover_finds_nifti_files_too(tmp_path, synthetic_nifti):
    path, _, _ = synthetic_nifti
    found = discover_scans(path.parent)
    assert [(e.scan_path, e.format) for e in found] == [(path.name, "nifti")]


def test_discover_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_scans(tmp_path / "nope")


def test_scan_ids_and_patient_keys():
    assert make_scan_id("4214-26/P00001/S0001") == "4214-26_P00001_S0001"
    assert make_scan_id("sub/train_1_a_1.nii.gz") == "sub_train_1_a_1"
    # P00001 repeats across top folders, so depth 1 (the top folder) is the patient:
    assert patient_key("4214-26/P00001/S0001", depth=1) == "4214-26"
    assert patient_key("4203-26/P00001/S0001", depth=1) != patient_key("4214-26/P00001/S0001", depth=1)
    assert patient_key("4214-26/P00001/S0001", depth=2) == "4214-26_P00001"
    assert patient_key("lonely.nii.gz") == "lonely"
