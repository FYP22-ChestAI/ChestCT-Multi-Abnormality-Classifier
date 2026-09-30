"""DICOM through the shared pipeline, the folder manifest, staging, the
raw-data stale-cache guard, and the scripts run for real."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chestct.preprocessing.config import DataConfig, PathsConfig, SourceConfig
from chestct.preprocessing.dicom_loader import discover_scans
from chestct.preprocessing.manifest import build_manifest, build_manifest_folder
from chestct.preprocessing.pipeline import process_scan
from chestct.preprocessing.preprocess import PreprocessConfig, is_cache_fresh, preprocess_one
from chestct.preprocessing.quality import QCThresholds
from chestct.preprocessing.staging import stage_tree
from chestct.inference import run_inference

REPO = Path(__file__).resolve().parents[1]
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
    [result] = run_inference(folder, cfg=CFG, qc_thresholds=RELAXED, normalize_imagenet=False)
    assert result.passed and result.pixel_values.shape == (40, 3, 48, 48)

    [rejected] = run_inference(folder, cfg=CFG)  # default thresholds want >= 80 slices
    assert not rejected.passed and rejected.pixel_values is None
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
    assert failed.pixel_values is None and failed.qc is not None


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
    m = build_manifest_folder(root, discover_scans(root), patient_depth=1)

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
    m = build_manifest(["train_1_a_1", "train_1_a_2"], None)
    assert list(m["volume_id"]) == ["train_1_a_1", "train_1_a_2"]
    assert list(m["scan_path"]) == ["train_1_a_1.nii.gz", "train_1_a_2.nii.gz"]
    assert set(m["format"]) == {"nifti"}


# ------------------------------------------------------------------ staging
def test_stage_tree_copies_resumes_and_filters(tmp_path, dicom_writer, synthetic_hu):
    src = _nhrd_tree(tmp_path / "drive", dicom_writer, synthetic_hu)
    dst = tmp_path / "local"

    first = stage_tree(src, dst, only_folders=["4214-26"], workers=2)
    assert first["copied"] == 6 and first["skipped"] == 0
    assert not (dst / "4203-26").exists()

    again = stage_tree(src, dst, only_folders=["4214-26"], workers=2)
    assert again["copied"] == 0 and again["skipped"] == 6  # resumable

    assert [e.scan_path for e in discover_scans(dst)] == ["4214-26/P00001/S0001"]
    with pytest.raises(FileNotFoundError):
        stage_tree(tmp_path / "missing", dst)


# ------------------------------------------------------------------- config
def test_per_source_qc_overrides():
    cfg = DataConfig(
        paths=PathsConfig(), preprocess=PreprocessConfig(), qc=QCThresholds(min_slices=80), windows={},
        sources={"local": SourceConfig(qc={"min_slices": 30})},
    )
    assert cfg.qc_for("local").min_slices == 30
    assert cfg.qc_for("local").max_slices == cfg.qc.max_slices  # everything else inherited
    assert cfg.qc_for("other").min_slices == 80 and cfg.qc_for(None).min_slices == 80


# ------------------------------------------------- the scripts, run for real
def _run(*args):
    proc = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout


def test_scripts_end_to_end_on_a_dicom_folder_given_at_run_time(tmp_path, dicom_writer, synthetic_hu):
    drive = _nhrd_tree(tmp_path / "some_mount", dicom_writer, synthetic_hu)
    work = tmp_path / "work"
    cfg_path = work / "data.yaml"
    work.mkdir()
    cfg_path.write_text(
        f"""
paths:
  raw_dir: {work / 'unused'}
  cache_dir: {work / 'cache'}
  manifest_path: {work / 'manifest.csv'}
  qc_report_path: {work / 'qc.csv'}
  montage_dir: {work / 'montages'}
  preprocessing_manifest_path: {work / 'pp.json'}
sources:
  nhrd_local:
    format: dicom
    raw_dir: {work / 'placeholder'}
    manifest_builder: folder
    patient_path_depth: 1
    qc: {{min_slices: 3}}
preprocess:
  target_size_hw: [32, 32]
"""
    )

    # the folder is only ever given on the command line; no absolute path is stored
    _run("scripts/build_manifest.py", "--config", cfg_path, "--source-name", "nhrd_local",
         "--raw-dir", drive, "--n-train", "2", "--n-val", "0", "--n-test", "0")
    manifest = pd.read_csv(work / "manifest.csv")
    assert len(manifest) == 2 and set(manifest["source_name"]) == {"nhrd_local"}
    assert set(manifest["split"]) == {"train"}
    assert "Cardiomegaly" not in manifest.columns  # no labels involved anywhere

    out = _run("scripts/preprocess_all.py", "--config", cfg_path, "--workers", 1,
               "--source-root", f"nhrd_local={drive}")
    assert "2 ok, 0 failed" in out
    assert len(list((work / "cache").glob("*.npy"))) == 2

    again = _run("scripts/preprocess_all.py", "--config", cfg_path, "--workers", 1,
                 "--source-root", f"nhrd_local={drive}")
    assert "0 volumes to process (2 already cached" in again

    _run("scripts/qc_report.py", "--config", cfg_path)
    qc = pd.read_csv(work / "qc.csv")
    assert len(qc) == 2 and qc["passed"].all()  # per-source min_slices=3 applied

    [sidecar_path] = (work / "cache").glob("4203-26_P00001_S0001*.meta.json")
    sidecar = json.loads(sidecar_path.read_text())
    assert "source_signature" in sidecar and "fingerprint" in sidecar
