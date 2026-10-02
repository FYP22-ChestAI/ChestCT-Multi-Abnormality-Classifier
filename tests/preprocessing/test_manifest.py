import pandas as pd
import pytest

from ct_preprocessing.manifest import (
    build_manifest_ctrate,
    check_no_patient_overlap,
    parse_volume_id,
)


def test_parse_volume_id():
    parsed = parse_volume_id("train_1_a_2")
    assert parsed == {
        "volume_id": "train_1_a_2",
        "patient_id": "train_1",
        "scan_id": "a",
        "reconstruction_id": 2,
        "source_split": "train",
    }


def test_parse_volume_id_rejects_bad_ids():
    with pytest.raises(ValueError):
        parse_volume_id("not_a_ctrate_id")


def test_reconstructions_of_same_scan_share_one_patient_id():
    a = parse_volume_id("train_5_a_1")
    b = parse_volume_id("train_5_a_2")
    assert a["patient_id"] == b["patient_id"]


def test_build_manifest_has_no_labels_and_no_split():
    manifest = build_manifest_ctrate(["train_1_a_1", "train_2_a_1"])
    assert "Cardiomegaly" not in manifest.columns
    assert not any("label" in c.lower() for c in manifest.columns)
    assert "split" not in manifest.columns  # splits are decided later, once, after QC
    assert list(manifest["source_split"]) == ["train", "train"]  # the OFFICIAL pool is kept, though


def test_build_manifest_joins_metadata_and_matches_nii_gz_extension():
    # Real CT-RATE CSVs store VolumeName WITH the extension, e.g. "train_1_a_1.nii.gz"
    metadata = pd.DataFrame(
        {"VolumeName": ["train_1_a_1.nii.gz", "train_2_a_1.nii.gz"], "Rows": [512, 512]}
    )
    manifest = build_manifest_ctrate(["train_1_a_1", "train_2_a_1"], metadata)
    assert list(manifest["Rows"]) == [512, 512]
    assert list(manifest["scan_path"]) == ["train_1_a_1.nii.gz", "train_2_a_1.nii.gz"]


def test_valid_and_train_patients_with_the_same_number_are_different_patients():
    manifest = build_manifest_ctrate(["train_1_a_1", "valid_1_a_1"])
    assert manifest["patient_id"].nunique() == 2


def test_check_no_patient_overlap_catches_a_leak():
    manifest = pd.DataFrame(
        {
            "volume_id": ["train_1_a_1", "train_1_a_2"],
            "patient_id": ["train_1", "train_1"],
            "split": ["train", "val"],  # same patient in two splits -- a leak
        }
    )
    with pytest.raises(ValueError):
        check_no_patient_overlap(manifest)


def test_check_no_patient_overlap_passes_clean_manifest():
    manifest = pd.DataFrame(
        {
            "volume_id": ["train_1_a_1", "train_2_a_1"],
            "patient_id": ["train_1", "train_2"],
            "split": ["train", "val"],
        }
    )
    check_no_patient_overlap(manifest)  # should not raise
