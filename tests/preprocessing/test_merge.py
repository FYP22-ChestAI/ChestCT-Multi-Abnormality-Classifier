"""Merging one run's chunk manifests into the run's manifest, plus the checklist and run record."""
import json

import pandas as pd
import pytest

from ct_preprocessing.config import DataConfig, PathsConfig, SourceConfig
from ct_preprocessing.ingest import state
from ct_preprocessing.ingest.labels import label_column
from ct_preprocessing.ingest.merge import (
    merge_run, read_chunk_manifests, read_patients_file, reconcile_patients, write_run_record,
)
from ct_preprocessing.ingest.splits import write_splits
from ct_preprocessing.preprocess import PreprocessConfig, config_fingerprint
from ct_preprocessing.quality import QCThresholds
from ct_preprocessing.runs import Run

TABLE = {("philips", "ya"): "sharp", ("philips", "b"): "soft", ("toshiba", "fc81"): "sharp"}


def make_cfg(tmp_path):
    d = tmp_path / "data"
    paths = PathsConfig(runs_dir=str(d / "runs"), state_dir=str(d / "state"), splits_dir=str(d / "splits"), cache_dir=str(d / "cache"))
    return DataConfig(
        paths=paths, preprocess=PreprocessConfig(), qc=QCThresholds(),
        sources={"ctrate": SourceConfig(manifest_builder="ctrate"),
                 "nhrd_local": SourceConfig(format="dicom", manifest_builder="folder")},
    )


def write_chunk(cfg, run, chunk_id, volume_ids, patients=None, failed=None, **columns):
    patients = patients or volume_ids
    rows = pd.DataFrame({
        "volume_id": volume_ids,
        "patient_id": patients,
        "scan_path": [f"{p}/S1" for p in patients],
        "source_name": run.source,
        "ingest_chunk": chunk_id,
        **columns,
    })
    f = run.chunk_manifest_path(chunk_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(f, index=False)
    state.write_done(run, chunk_id, {"n_failed": len(failed or {}), "failed": failed or {}})


def merge(cfg, run, **kw):
    kw.setdefault("kernel_table", TABLE)
    return merge_run(cfg, run, **kw)


def test_chunks_are_combined_one_row_per_volume(tmp_path):
    cfg = make_cfg(tmp_path)
    run = Run(cfg.paths, "ctrate", "train-sharp")
    write_chunk(cfg, run, "0000", ["train_1_a_1", "train_2_a_1"])
    write_chunk(cfg, run, "0001", ["train_3_a_1"])

    merged, report = merge(cfg, run)

    assert len(merged) == 3 and report.rows == 3
    assert merged.ingest_chunk.tolist() == ["0000", "0000", "0001"]  # zero padding survives
    assert set(merged.split) == {"unassigned"}  # nothing is split until assign_splits.py runs


def test_runs_are_merged_separately_and_do_not_see_each_others_chunks(tmp_path):
    cfg = make_cfg(tmp_path)
    sharp, soft = Run(cfg.paths, "ctrate", "train-sharp"), Run(cfg.paths, "ctrate", "train-soft")
    write_chunk(cfg, sharp, "0000", ["train_1_a_1"])
    write_chunk(cfg, soft, "0000", ["train_1_a_2", "train_2_a_2"])
    assert len(merge(cfg, sharp)[0]) == 1 and len(merge(cfg, soft)[0]) == 2
    assert len(read_chunk_manifests(sharp)) == 1


def test_a_scan_that_arrived_twice_is_kept_once(tmp_path):
    cfg = make_cfg(tmp_path)
    run = Run(cfg.paths, "nhrd_local", "main")
    write_chunk(cfg, run, "nhrd_A_001", ["v1", "v2"])
    write_chunk(cfg, run, "nhrd_B_001", ["v2", "v3"])  # patient uploaded in two archives by mistake
    merged, report = merge(cfg, run)
    assert sorted(merged.volume_id) == ["v1", "v2", "v3"] and report.duplicates_dropped == 1
    assert merged.set_index("volume_id").loc["v2", "ingest_chunk"] == "nhrd_A_001"  # the first one wins


def test_splits_already_frozen_are_joined_back_in_and_shared_by_every_run(tmp_path):
    cfg = make_cfg(tmp_path)
    write_splits(state.splits_path(cfg.paths, "ctrate"), {"p1": "train", "p2": "val"})
    for name in ("train-sharp", "train-soft"):
        run = Run(cfg.paths, "ctrate", name)
        write_chunk(cfg, run, "0000", ["v1", "v2", "v3"], patients=["p1", "p2", "p3"])
        merged, _ = merge(cfg, run)
        assert merged.set_index("patient_id").split.to_dict() == {"p1": "train", "p2": "val", "p3": "unassigned"}


def test_merging_a_run_with_nothing_ingested_says_what_to_do(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(ValueError, match="ingest.py --run train-sharp"):
        merge(cfg, Run(cfg.paths, "ctrate", "train-sharp"))


def test_failed_volumes_from_the_done_markers_are_reported(tmp_path):
    cfg = make_cfg(tmp_path)
    run = Run(cfg.paths, "ctrate", "train-sharp")
    write_chunk(cfg, run, "0000", ["train_1_a_1", "train_2_a_1"], failed={"train_2_a_1": "corrupt"})
    _, report = merge(cfg, run)
    assert report.failures == [{"source_name": "ctrate", "ingest_chunk": "0000", "volume_id": "train_2_a_1", "error": "corrupt"}]


def test_the_run_record_holds_the_cache_fingerprint_the_run_and_the_failures(tmp_path):
    cfg = make_cfg(tmp_path)
    run = Run(cfg.paths, "ctrate", "train-sharp")
    write_chunk(cfg, run, "0000", ["train_1_a_1", "train_2_a_1"], failed={"train_2_a_1": "corrupt"},
                Manufacturer="Philips", ConvolutionKernel="YA")
    merged, report = merge(cfg, run)
    write_run_record(run, cfg, merged, report)
    rec = json.loads(run.record_path.read_text())
    assert rec["fingerprint"] == config_fingerprint(cfg.preprocess)
    assert rec["run"] == "train-sharp" and rec["source"] == "ctrate"
    assert rec["n_volumes"] == 2 and rec["n_failed_volumes"] == 1 and rec["kernel_classes"] == {"sharp": 2}
    assert rec["preprocess_config"]["target_size_hw"] == [224, 224]


# ----------------------------------------------------------------------- kernel columns
def test_ctrate_rows_get_a_normalised_kernel_and_a_class(tmp_path):
    cfg = make_cfg(tmp_path)
    run = Run(cfg.paths, "ctrate", "train-sharp")
    write_chunk(cfg, run, "0000", ["train_1_a_1", "train_1_a_2", "train_2_a_1"], patients=["train_1", "train_1", "train_2"],
                Manufacturer=["Philips", "Philips", "Mystery"], ConvolutionKernel=["YA", "['B', '3']", "ZZ"])
    merged, report = merge(cfg, run)
    assert merged.kernel.tolist() == ["YA", "B", "ZZ"]
    assert merged.kernel_class.tolist() == ["sharp", "soft", "other"]
    assert "manufacturer" in merged.columns and "Manufacturer" not in merged.columns
    assert report.kernel_classes == {"sharp": 1, "soft": 1, "other": 1}


def test_dicom_rows_keep_their_kernel_and_get_a_class(tmp_path):
    cfg = make_cfg(tmp_path)
    run = Run(cfg.paths, "nhrd_local", "main")
    write_chunk(cfg, run, "a", ["v1", "v2"], manufacturer=["TOSHIBA", "TOSHIBA"], kernel=["FC81", "FC03"])
    merged, _ = merge(cfg, run)
    assert merged.kernel_class.tolist() == ["sharp", "other"]  # FC03 is not in the table


def test_a_run_without_scanner_columns_still_merges_as_other(tmp_path):
    cfg = make_cfg(tmp_path)
    run = Run(cfg.paths, "ctrate", "train-sharp")
    write_chunk(cfg, run, "0000", ["train_1_a_1"])
    merged, _ = merge(cfg, run)
    assert merged.kernel_class.tolist() == ["other"]


# ---------------------------------------------------------------------------- labels
def test_ctrate_labels_are_joined_by_volume_id(tmp_path):
    cfg = make_cfg(tmp_path)
    run = Run(cfg.paths, "ctrate", "train-sharp")
    write_chunk(cfg, run, "0000", ["train_1_a_1", "train_2_a_1"])
    labels = pd.DataFrame({"volume_id": ["train_1_a_1", "train_2_a_1", "train_9_a_1"],
                           "Lung nodule": [1, 0, 1], "Medical material": [0, 1, 1]})
    merged, report = merge(cfg, run, labels=labels)
    assert merged[label_column("Lung nodule")].tolist() == [1, 0] and merged["label_medical_material"].tolist() == [0, 1]
    assert report.labels.rows_with_labels == 2 and report.labels.unmatched_label_keys == 1


def test_archive_labels_are_joined_by_scan_path_and_unlabelled_scans_stay_empty(tmp_path):
    cfg = make_cfg(tmp_path)
    run = Run(cfg.paths, "nhrd_local", "main")
    write_chunk(cfg, run, "a", ["v1", "v2"], patients=["4203-26", "4213-26"])
    labels = pd.DataFrame({"scan_path": ["4203-26/S1/"], "Pleural effusion": [1]})
    merged, report = merge(cfg, run, labels=labels, manifest_key="scan_path", labels_key="scan_path")
    assert merged.label_pleural_effusion.tolist()[0] == 1 and pd.isna(merged.label_pleural_effusion.tolist()[1])
    assert report.labels.rows_with_labels == 1 and "1 of 2 manifest rows labelled" in report.labels.format()


def test_a_missing_key_column_is_reported_clearly(tmp_path):
    cfg = make_cfg(tmp_path)
    run = Run(cfg.paths, "ctrate", "train-sharp")
    write_chunk(cfg, run, "0000", ["train_1_a_1"])
    with pytest.raises(ValueError, match="no 'patient_key' column"):
        merge(cfg, run, labels=pd.DataFrame({"x": [1], "y": [2]}), labels_key="patient_key")


# ------------------------------------------------------------------- the checklist
def test_the_patients_file_ignores_blank_lines_comments_and_trailing_slashes(tmp_path):
    f = tmp_path / "patients.txt"
    f.write_text("# exported from the portable disk\n4203-26\n\n4214-26/\n  4301-11  \n")
    assert read_patients_file(f) == ["4203-26", "4214-26", "4301-11"]


def test_reconciliation_reports_patients_that_never_arrived_and_unexpected_ones():
    rows = pd.DataFrame({"scan_path": ["4203-26/P1/S1", "4203-26/P2/S1", "9999-99/P1/S1"]})
    missing, unexpected = reconcile_patients(rows, ["4203-26", "4214-26", "4301-11"])
    assert missing == ["4214-26", "4301-11"] and unexpected == ["9999-99"]


# ------------------------------------------------------- the oversized-patient warning
def test_an_archive_zipped_one_folder_too_high_shows_up_as_one_giant_patient():
    from ct_preprocessing.ingest.merge import suspicious_patients

    normal = pd.DataFrame({"patient_id": [f"p{i}" for i in range(30) for _ in range(2)]})  # 30 patients x 2 scans
    giant = pd.DataFrame({"patient_id": ["NHRD - CT Scan"] * 60})  # everything under one top folder
    assert suspicious_patients(normal) == []
    assert suspicious_patients(pd.concat([normal, giant])) == [("NHRD - CT Scan", 60)]


def test_a_patient_with_a_handful_of_scans_is_not_suspicious():
    from ct_preprocessing.ingest.merge import suspicious_patients

    rows = pd.DataFrame({"patient_id": ["a"] * 6 + ["b"] * 1 + ["c"] * 1})
    assert suspicious_patients(rows) == []
    assert suspicious_patients(pd.DataFrame({"patient_id": []})) == []
