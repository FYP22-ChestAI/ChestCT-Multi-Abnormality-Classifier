"""Merging per-chunk manifests into one manifest, plus the checklist and run record."""
import json

import pandas as pd
import pytest

from ct_preprocessing.config import DataConfig, PathsConfig, SourceConfig
from ct_preprocessing.ingest import state
from ct_preprocessing.ingest.merge import (
    merge_sources, read_chunk_manifests, read_patients_file, reconcile_patients, write_run_record,
)
from ct_preprocessing.ingest.splits import write_splits
from ct_preprocessing.preprocess import PreprocessConfig, config_fingerprint
from ct_preprocessing.quality import QCThresholds


def make_cfg(tmp_path):
    d = tmp_path / "data"
    paths = PathsConfig(chunk_manifest_dir=str(d / "manifests"), state_dir=str(d / "state"), splits_dir=str(d / "splits"),
                        manifest_path=str(d / "manifest.csv"))
    return DataConfig(
        paths=paths, preprocess=PreprocessConfig(), qc=QCThresholds(),
        sources={"ctrate": SourceConfig(manifest_builder="ctrate"),
                 "nhrd_local": SourceConfig(format="dicom", manifest_builder="folder")},
    )


def write_chunk(cfg, source, chunk_id, volume_ids, patients=None, failed=None):
    rows = pd.DataFrame({
        "volume_id": volume_ids,
        "patient_id": patients or volume_ids,
        "scan_path": [f"{(patients or volume_ids)[i]}/S1" for i in range(len(volume_ids))],
        "source_name": source,
        "ingest_chunk": chunk_id,
    })
    f = state.chunk_manifest_path(cfg.paths, source, chunk_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(f, index=False)
    state.write_done(cfg.paths, source, chunk_id, {"n_failed": len(failed or {}), "failed": failed or {}})


def test_chunks_of_every_source_are_combined_one_row_per_volume(tmp_path):
    cfg = make_cfg(tmp_path)
    write_chunk(cfg, "ctrate", "0000", ["train_1_a_1", "train_2_a_1"])
    write_chunk(cfg, "ctrate", "0001", ["train_3_a_1"])
    write_chunk(cfg, "nhrd_local", "nhrd_A_001", ["4203-26_S1"])

    merged, report = merge_sources(cfg, ["ctrate", "nhrd_local"])

    assert len(merged) == 4 and report.rows_per_source == {"ctrate": 3, "nhrd_local": 1}
    assert merged.ingest_chunk.tolist() == ["0000", "0000", "0001", "nhrd_A_001"]  # zero padding survives
    assert set(merged.split) == {"unassigned"}  # nothing is split until assign_splits.py runs


def test_a_scan_that_arrived_twice_is_kept_once(tmp_path):
    cfg = make_cfg(tmp_path)
    write_chunk(cfg, "nhrd_local", "nhrd_A_001", ["v1", "v2"])
    write_chunk(cfg, "nhrd_local", "nhrd_B_001", ["v2", "v3"])  # patient uploaded in two archives by mistake
    merged, report = merge_sources(cfg, ["nhrd_local"])
    assert sorted(merged.volume_id) == ["v1", "v2", "v3"] and report.duplicates_dropped == 1
    assert merged.set_index("volume_id").loc["v2", "ingest_chunk"] == "nhrd_A_001"  # the first one wins


def test_the_same_volume_id_in_two_sources_is_an_error(tmp_path):
    cfg = make_cfg(tmp_path)
    write_chunk(cfg, "ctrate", "0000", ["same"])
    write_chunk(cfg, "nhrd_local", "a", ["same"])
    with pytest.raises(ValueError, match="not unique across sources"):
        merge_sources(cfg, ["ctrate", "nhrd_local"])


def test_splits_already_frozen_are_joined_back_in_on_every_merge(tmp_path):
    cfg = make_cfg(tmp_path)
    write_chunk(cfg, "nhrd_local", "a", ["v1", "v2", "v3"], patients=["p1", "p2", "p3"])
    write_splits(state.splits_path(cfg.paths, "nhrd_local"), {"p1": "train", "p2": "val"})
    merged, _ = merge_sources(cfg, ["nhrd_local"])
    assert merged.set_index("patient_id").split.to_dict() == {"p1": "train", "p2": "val", "p3": "unassigned"}


def test_merging_with_nothing_ingested_says_what_to_do(tmp_path):
    with pytest.raises(ValueError, match="run ingest.py"):
        merge_sources(make_cfg(tmp_path), ["ctrate"])


def test_sources_without_chunks_are_skipped(tmp_path):
    cfg = make_cfg(tmp_path)
    write_chunk(cfg, "ctrate", "0000", ["train_1_a_1"])
    merged, report = merge_sources(cfg, ["ctrate", "nhrd_local"])
    assert list(report.rows_per_source) == ["ctrate"] and len(merged) == 1
    assert read_chunk_manifests(cfg, "nhrd_local").empty


def test_failed_volumes_from_the_done_markers_are_reported(tmp_path):
    cfg = make_cfg(tmp_path)
    write_chunk(cfg, "ctrate", "0000", ["train_1_a_1", "train_2_a_1"], failed={"train_2_a_1": "corrupt"})
    _, report = merge_sources(cfg, ["ctrate"])
    assert report.failures == [{"source_name": "ctrate", "ingest_chunk": "0000", "volume_id": "train_2_a_1", "error": "corrupt"}]


def test_the_run_record_holds_the_cache_fingerprint_and_the_failures(tmp_path):
    cfg = make_cfg(tmp_path)
    write_chunk(cfg, "ctrate", "0000", ["train_1_a_1", "train_2_a_1"], failed={"train_2_a_1": "corrupt"})
    merged, report = merge_sources(cfg, ["ctrate"])
    write_run_record(tmp_path / "rec.json", cfg, merged, report)
    rec = json.loads((tmp_path / "rec.json").read_text())
    assert rec["fingerprint"] == config_fingerprint(cfg.preprocess)
    assert rec["n_volumes"] == 2 and rec["n_failed_volumes"] == 1 and rec["volumes_per_source"] == {"ctrate": 2}
    assert rec["preprocess_config"]["target_size_hw"] == [224, 224]


# ----------------------------------------------------------------- the checklist
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
