"""DICOM through the shared pipeline, the folder manifest, the raw-data
stale-cache guard and per-source QC overrides."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ct_preprocessing.config import DataConfig, PathsConfig, SourceConfig
from ct_preprocessing.dicom_loader import discover_scans
from ct_preprocessing.manifest import build_manifest_ctrate, build_manifest_folder
from ct_preprocessing.pipeline import process_scan
from ct_preprocessing.preprocess import PreprocessConfig, is_cache_fresh, preprocess_one
from ct_preprocessing.quality import QCThresholds
from ct_preprocessing.inference import run_inference

CFG = PreprocessConfig(target_size_hw=(48, 48))
RELAXED = QCThresholds(min_slices=10)
NIFTI_ORIENTATION = (-1, 0, 0, 0, -1, 0)  # DICOM orientation matching the NIfTI fixture's x->R, y->A


def test_dicom_and_nifti_of_the_same_volume_give_identical_pipeline_output(tmp_path, dicom_writer, synthetic_nifti):
    nifti_path, hu, _ = synthetic_nifti
    folder = tmp_path / "S0001"
    dicom_writer(folder, hu, spacing_zyx=(1.5, 1.0, 1.0), iop=NIFTI_ORIENTATION)

    a = process_scan(nifti_path, CFG, qc_thresholds=RELAXED)
    b = process_scan(folder, CFG, qc_thresholds=RELAXED)

    assert a.error is None and b.error is None
    assert b.meta["format"] == "dicom" and a.meta["format"] == "nifti"
    np.testing.assert_array_equal(a.hu, b.hu)
    assert b.qc.passed


def test_missing_slice_fails_qc_with_a_clear_reason(tmp_path, dicom_writer, synthetic_hu):
    folder = tmp_path / "S0001"
    dicom_writer(folder, synthetic_hu, skip_slices=(10,))
    result = process_scan(folder, CFG, qc_thresholds=RELAXED)
    assert result.hu is not None  # still returned; the caller decides
    assert not result.qc.passed
    assert any("non-uniform slice spacing" in r for r in result.qc.reasons)


def test_a_folder_with_two_series_is_a_hard_error(tmp_path, dicom_writer, synthetic_hu):
    folder = tmp_path / "S0001"
    dicom_writer(folder, synthetic_hu, name_prefix="A")
    dicom_writer(folder, synthetic_hu, name_prefix="B")
    result = process_scan(folder, CFG, qc_thresholds=RELAXED)
    assert result.error and "different DICOM series" in result.error


def test_inference_takes_a_dicom_folder(tmp_path, dicom_writer, synthetic_hu):
    folder = tmp_path / "S0001"
    dicom_writer(folder, synthetic_hu, iop=NIFTI_ORIENTATION)
    [result] = run_inference(folder, cfg=CFG, qc_thresholds=RELAXED)
    assert result.passed and result.hu.shape == (40, 48, 48)

    [rejected] = run_inference(folder, cfg=CFG)  # default thresholds want >= 80 slices
    assert not rejected.passed and rejected.hu is None
    assert rejected.qc is not None and not rejected.qc.passed


def test_inference_batch_continues_past_a_qc_failure(tmp_path, dicom_writer, synthetic_hu):
    # A folder holding several scans (e.g. several patients) -- one fails QC,
    # the rest must still be processed rather than the whole request dying.
    dicom_writer(tmp_path / "good" / "S0001", synthetic_hu[:10], iop=NIFTI_ORIENTATION)
    dicom_writer(tmp_path / "bad" / "S0001", synthetic_hu[:6], iop=NIFTI_ORIENTATION)
    results = run_inference(tmp_path, cfg=CFG, qc_thresholds=RELAXED)
    assert len(results) == 2
    assert {r.passed for r in results} == {True, False}
    failed = next(r for r in results if not r.passed)
    assert failed.hu is None and failed.qc is not None


# ------------------------------------------------------------ stale-cache guard
def test_cache_is_stale_when_the_raw_data_changes_under_the_same_id(tmp_path, dicom_writer, synthetic_hu):
    folder, cache = tmp_path / "S0001", tmp_path / "cache"
    dicom_writer(folder, synthetic_hu, iop=NIFTI_ORIENTATION)
    assert preprocess_one(folder, cache, CFG, volume_id="scan1").ok
    assert is_cache_fresh(cache, "scan1", CFG, scan_path=folder)

    dicom_writer(folder, synthetic_hu[:30], iop=NIFTI_ORIENTATION, name_prefix="NEW")  # different files appear
    assert not is_cache_fresh(cache, "scan1", CFG, scan_path=folder)


def test_cache_is_stale_when_a_nifti_is_replaced_by_a_different_download(tmp_path, synthetic_nifti):
    import nibabel as nib

    path, _, _ = synthetic_nifti
    cache = tmp_path / "cache"
    assert preprocess_one(path, cache, CFG, volume_id="v").ok
    assert is_cache_fresh(cache, "v", CFG, scan_path=path)

    img = nib.load(str(path))  # "re-download": same name, different content
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj)[:, :, :30] * 1.0, img.affine), str(path))
    assert not is_cache_fresh(cache, "v", CFG, scan_path=path)


# ---------------------------------------------------------------- manifests
def _nhrd_tree(root, dicom_writer, hu):
    for top in ("4203-26", "4214-26"):
        dicom_writer(root / top / "P00001" / "S0001", hu[:6], patient_id=f"anon-{top}")
    return root


def test_folder_manifest_has_no_labels_and_patients_do_not_collide(tmp_path, dicom_writer, synthetic_hu):
    root = _nhrd_tree(tmp_path, dicom_writer, synthetic_hu)
    # patient_id_source="path" pinned explicitly: this test is about folder-depth
    # grouping specifically, not the "auto" default's own decision (covered below) --
    # _nhrd_tree's fixture happens to give each folder a distinct PatientID tag too,
    # which "auto" would otherwise (correctly) prefer.
    m = build_manifest_folder(root, discover_scans(root), patient_depth=1, patient_id_source="path")

    assert m["volume_id"].iloc[0].startswith("4203-26_P00001_S0001")
    assert m["volume_id"].iloc[1].startswith("4214-26_P00001_S0001")
    assert m["volume_id"].is_unique
    assert list(m["patient_id"]) == ["4203-26", "4214-26"]  # 'P00001' repeats, the top folder does not
    assert "split" not in m.columns  # split assignment is the caller's job, not this function's
    assert set(m["format"]) == {"dicom"}
    assert list(m["scan_path"]) == ["4203-26/P00001/S0001", "4214-26/P00001/S0001"]
    assert m["manufacturer"].iloc[0] == "TESTCO" and m["kernel"].iloc[0] == "B70f"
    assert not any(c.lower().startswith(("patientid", "patient_name")) for c in m.columns)
    assert m["scan_path"].map(lambda p: not Path(p).is_absolute()).all()


def test_folder_manifest_can_group_patients_by_dicom_tag(tmp_path, dicom_writer, synthetic_hu):
    root = _nhrd_tree(tmp_path, dicom_writer, synthetic_hu)
    m = build_manifest_folder(root, discover_scans(root), patient_id_source="dicom_tag")
    assert m["patient_id"].str.startswith("dicom_").all() and m["patient_id"].nunique() == 2


def test_folder_manifest_auto_prefers_dicom_tag_when_present_and_distinct(tmp_path, dicom_writer, synthetic_hu, capsys):
    # _nhrd_tree's fixture gives each top folder its own distinct PatientID tag --
    # "auto" (the default) should confirm that and use it, same as explicit "dicom_tag".
    root = _nhrd_tree(tmp_path, dicom_writer, synthetic_hu)
    m = build_manifest_folder(root, discover_scans(root), patient_depth=1)
    assert m["patient_id"].str.startswith("dicom_").all() and m["patient_id"].nunique() == 2
    assert "using patient_id_source=dicom_tag" in capsys.readouterr().out


def test_folder_manifest_auto_falls_back_to_path_when_patient_id_is_missing(tmp_path, dicom_writer, synthetic_hu, capsys):
    # write_dicom_series' default patient_id ("ANON01") is the SAME for every
    # call unless overridden -- simulating anonymised real data where the tag
    # survives but no longer distinguishes different patients.
    folder = tmp_path / "x"
    dicom_writer(folder / "4203-26" / "P00001" / "S0001", synthetic_hu[:6])
    dicom_writer(folder / "4214-26" / "P00001" / "S0001", synthetic_hu[:6])
    m = build_manifest_folder(folder, discover_scans(folder), patient_depth=1)
    assert list(m["patient_id"]) == ["4203-26", "4214-26"]
    assert "using patient_id_source=path" in capsys.readouterr().out


def test_folder_manifest_splits_a_mixed_series_folder_into_two_rows(tmp_path, dicom_writer, synthetic_hu):
    folder = tmp_path / "x" / "S0001"
    dicom_writer(folder, synthetic_hu[:6], name_prefix="A")
    dicom_writer(folder, synthetic_hu[:6], name_prefix="B")
    m = build_manifest_folder(tmp_path, discover_scans(tmp_path))
    assert len(m) == 2
    assert list(m["scan_path"]) == ["x/S0001", "x/S0001"]
    assert m["series_uid"].nunique() == 2
    assert m["volume_id"].is_unique
    assert "manifest_problem" not in m.columns or m["manifest_problem"].isna().all()


def test_ctrate_manifest_works_without_labels():
    m = build_manifest_ctrate(["train_1_a_1", "train_1_a_2"], None)
    assert list(m["volume_id"]) == ["train_1_a_1", "train_1_a_2"]
    assert list(m["scan_path"]) == ["train_1_a_1.nii.gz", "train_1_a_2.nii.gz"]
    assert set(m["format"]) == {"nifti"}


# ------------------------------------------------------------------- config
def test_per_source_qc_overrides():
    cfg = DataConfig(
        paths=PathsConfig(), preprocess=PreprocessConfig(), qc=QCThresholds(min_slices=80),
        sources={"local": SourceConfig(qc={"min_slices": 30})},
    )
    assert cfg.qc_for("local").min_slices == 30
    assert cfg.qc_for("local").max_slices == cfg.qc.max_slices  # everything else inherited
    assert cfg.qc_for("other").min_slices == 80 and cfg.qc_for(None).min_slices == 80
