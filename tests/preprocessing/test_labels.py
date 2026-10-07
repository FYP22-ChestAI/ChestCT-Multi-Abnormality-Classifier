"""Labels: downloaded for CT-RATE, read from the Drive folder for archive sources, joined at merge time only."""
import pandas as pd
import pytest

from ct_preprocessing.config import LabelsConfig, PathsConfig
from ct_preprocessing.ingest.archives import LocalBackend
from ct_preprocessing.ingest.base import FetchError
from ct_preprocessing.ingest.labels import (
    CTRATE_LABEL_FILES, attach_labels, download_ctrate_labels, fetch_archive_labels, label_column,
    load_ctrate_labels, read_labels_csv,
)


def test_label_columns_are_snake_case_with_a_prefix():
    assert label_column("Medical material") == "label_medical_material"
    assert label_column("Coronary artery wall calcification") == "label_coronary_artery_wall_calcification"
    assert label_column(" Pleural  effusion ") == "label_pleural_effusion"


def _fake_hf(tmp_path, fail=()):
    calls = []

    def hf(repo_id, filename, repo_type):
        calls.append(filename)
        if filename in fail:
            raise OSError("404 not found")
        src = tmp_path / f"hf_{len(calls)}.csv"
        pool = "train" if "train" in filename else "valid"
        src.write_text(f"VolumeName,Lung nodule\n{pool}_1_a_1.nii.gz,1\n{pool}_2_a_1.nii.gz,0\n")
        return str(src)

    return hf, calls


def test_the_ctrate_label_files_are_downloaded_next_to_the_metadata_and_loaded_by_volume_id(tmp_path):
    paths = PathsConfig(metadata_dir=str(tmp_path / "meta"))
    hf, calls = _fake_hf(tmp_path)
    assert download_ctrate_labels(paths, "org/CT-RATE", hf) == []
    assert sorted(calls) == sorted(CTRATE_LABEL_FILES.values())
    labels = load_ctrate_labels(paths)
    assert sorted(labels.volume_id) == ["train_1_a_1", "train_2_a_1", "valid_1_a_1", "valid_2_a_1"]
    download_ctrate_labels(paths, "org/CT-RATE", hf)  # present already: nothing is fetched again
    assert len(calls) == 2
    download_ctrate_labels(paths, "org/CT-RATE", hf, refresh=True)
    assert len(calls) == 4


def test_labels_are_optional_a_failed_download_is_a_warning_not_an_error(tmp_path):
    paths = PathsConfig(metadata_dir=str(tmp_path / "meta"))
    hf, _ = _fake_hf(tmp_path, fail={CTRATE_LABEL_FILES["train"]})
    warnings = download_ctrate_labels(paths, "org/CT-RATE", hf)
    assert len(warnings) == 1 and "by hand" in warnings[0] and "train_predicted_labels.csv" in warnings[0]
    assert sorted(load_ctrate_labels(paths).volume_id) == ["valid_1_a_1", "valid_2_a_1"]  # the other pool still loads


def test_no_label_files_means_none(tmp_path):
    assert load_ctrate_labels(PathsConfig(metadata_dir=str(tmp_path / "empty"))) is None


def test_the_archive_labels_file_is_copied_out_of_the_drive_folder_and_read(tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    (drive / "labels.csv").write_text("scan_path,Lung nodule,Emphysema\n4203-26/P1/S1,1,0\n")
    (drive / "nhrd_A_001.zip").write_bytes(b"x")
    path = fetch_archive_labels(LocalBackend(drive), LabelsConfig(file="labels.csv"), tmp_path / "meta" / "n_labels.csv")
    frame = read_labels_csv(path)
    assert list(frame.columns) == ["scan_path", "Lung nodule", "Emphysema"] and frame.iloc[0, 0] == "4203-26/P1/S1"


def test_asking_for_labels_when_none_are_configured_or_the_file_is_missing_is_an_error(tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    with pytest.raises(ValueError, match="no labels file configured"):
        fetch_archive_labels(LocalBackend(drive), LabelsConfig(), tmp_path / "x.csv")
    with pytest.raises((FetchError, OSError)):
        fetch_archive_labels(LocalBackend(drive), LabelsConfig(file="labels.csv"), tmp_path / "x.csv")


def test_a_labels_file_needs_a_key_and_at_least_one_label(tmp_path):
    f = tmp_path / "l.csv"
    f.write_text("scan_path\n4203-26/P1/S1\n")
    with pytest.raises(ValueError, match="key column followed by at least one label"):
        read_labels_csv(f)


def test_labels_attach_by_key_and_the_report_counts_matches():
    manifest = pd.DataFrame({"volume_id": ["a", "b", "c"], "patient_id": ["p1", "p1", "p2"]})
    labels = pd.DataFrame({"volume_id": ["a", "c", "z"], "Emphysema": [1, 0, 1], "Lung nodule": [0, 1, 1]})
    out, report = attach_labels(manifest, labels, manifest_key="volume_id", labels_key="volume_id")
    assert out.label_emphysema.tolist()[0] == 1 and pd.isna(out.label_emphysema.tolist()[1]) and out.label_lung_nodule.tolist()[2] == 1
    assert report.columns == ["label_emphysema", "label_lung_nodule"]
    assert (report.rows, report.rows_with_labels, report.unmatched_label_keys) == (3, 2, 1)
    assert list(manifest.columns) == ["volume_id", "patient_id"]  # the input is not modified


def test_duplicate_label_keys_keep_the_first_and_are_counted():
    manifest = pd.DataFrame({"patient_id": ["p1", "p2"]})
    labels = pd.DataFrame({"patient_id": ["p1", "p1", "p2"], "Emphysema": [1, 0, 0]})
    out, report = attach_labels(manifest, labels, manifest_key="patient_id", labels_key="patient_id")
    assert out.label_emphysema.tolist() == [1, 0] and report.duplicate_label_keys == 1
    assert "duplicate label key" in report.format()


def test_attaching_twice_replaces_the_label_columns_instead_of_duplicating_them():
    manifest = pd.DataFrame({"volume_id": ["a"]})
    first, _ = attach_labels(manifest, pd.DataFrame({"volume_id": ["a"], "X": [1]}), manifest_key="volume_id", labels_key="volume_id")
    second, _ = attach_labels(first, pd.DataFrame({"volume_id": ["a"], "X": [0]}), manifest_key="volume_id", labels_key="volume_id")
    assert list(second.columns) == ["volume_id", "label_x"] and second.label_x.tolist() == [0]
