import pandas as pd
import pytest

from chestct.preprocessing.manifest import (
    assign_patient_splits,
    assign_splits_by_amount,
    build_manifest,
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


def test_build_manifest_has_no_labels():
    manifest = build_manifest(["train_1_a_1", "train_2_a_1"])
    assert "Cardiomegaly" not in manifest.columns
    assert not any("label" in c.lower() for c in manifest.columns)


def test_build_manifest_joins_metadata_and_matches_nii_gz_extension():
    # Real CT-RATE CSVs store VolumeName WITH the extension, e.g. "train_1_a_1.nii.gz"
    metadata = pd.DataFrame(
        {"VolumeName": ["train_1_a_1.nii.gz", "train_2_a_1.nii.gz"], "Rows": [512, 512]}
    )
    manifest = build_manifest(["train_1_a_1", "train_2_a_1"], metadata)
    assert list(manifest["Rows"]) == [512, 512]


def test_assign_splits_by_amount_is_exact_and_disjoint():
    patients = [f"train_{i}" for i in range(20)]
    split_map = assign_splits_by_amount(patients, n_train=10, n_val=5, n_test=3, seed=0)
    counts = pd.Series(split_map).value_counts()
    assert counts.get("train", 0) == 10
    assert counts.get("val", 0) == 5
    assert counts.get("test", 0) == 3
    assert len(split_map) == 18  # 2 patients left unselected, not forced into a split


def test_assign_splits_by_amount_rejects_asking_for_too_many():
    with pytest.raises(ValueError):
        assign_splits_by_amount(["train_1", "train_2"], n_train=2, n_val=1, n_test=0, seed=0)


def test_assign_patient_splits_is_disjoint_and_roughly_sized():
    patients = [f"train_{i}" for i in range(100)]
    split_map = assign_patient_splits(patients, val_fraction=0.1, seed=0)
    counts = pd.Series(split_map).value_counts()
    assert set(split_map.values()) <= {"train", "val"}
    assert 5 <= counts.get("val", 0) <= 15


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
