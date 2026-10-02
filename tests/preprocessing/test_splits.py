"""Patient-level, frozen, QC-aware train/val/test assignment."""
import pandas as pd
import pytest

from ct_preprocessing.config import SourceConfig, SplitConfig
from ct_preprocessing.ingest.splits import (
    InsufficientPatients, apply_splits, assign_source_splits, plan_split, read_splits, write_splits,
)

PATIENTS = [f"p{i:02d}" for i in range(20)]


def _counts(mapping):
    return pd.Series(mapping).value_counts().to_dict()


# ------------------------------------------------------------------ plan_split
def test_plan_split_hits_the_exact_counts_and_covers_everyone_once():
    m = plan_split(PATIENTS, n_val=3, n_test=2, seed=0)
    assert _counts(m) == {"train": 15, "val": 3, "test": 2}
    assert set(m) == set(PATIENTS)


def test_plan_split_is_deterministic_for_a_seed():
    assert plan_split(PATIENTS, n_val=3, n_test=2, seed=7) == plan_split(list(reversed(PATIENTS)), n_val=3, n_test=2, seed=7)
    assert plan_split(PATIENTS, n_val=3, n_test=2, seed=7) != plan_split(PATIENTS, n_val=3, n_test=2, seed=8)


def test_existing_assignments_are_never_changed_by_a_later_run():
    first = plan_split(PATIENTS[:12], n_val=2, n_test=2, seed=0)
    again = plan_split(PATIENTS, n_val=2, n_test=2, seed=99, existing=first)  # even with a different seed
    assert all(again[p] == s for p, s in first.items())
    assert set(again) == set(PATIENTS)
    assert all(again[p] == "train" for p in PATIENTS[12:])  # quotas were already full -> new patients are train


def test_a_top_up_only_fills_the_quota_that_is_still_missing():
    first = plan_split(PATIENTS[:6], n_val=1, n_test=0, seed=0)  # only 1 of the 3 wanted val patients so far
    again = plan_split(PATIENTS, n_val=3, n_test=0, seed=0, existing=first)
    assert list(again.values()).count("val") == 3
    assert all(again[p] == s for p, s in first.items())


def test_forced_rules_apply_and_do_not_count_against_the_quotas():
    forced = {p: "test" for p in PATIENTS[:5]}  # e.g. CT-RATE's official valid pool
    m = plan_split(PATIENTS, n_val=4, n_test=0, seed=0, forced=forced)
    assert all(m[p] == "test" for p in PATIENTS[:5])
    assert _counts(m) == {"train": 11, "val": 4, "test": 5}
    assert all(m[p] != "val" for p in PATIENTS[:5])  # val is drawn from the rest only


def test_asking_for_more_patients_than_exist_is_an_error_not_a_silent_shortfall():
    with pytest.raises(InsufficientPatients):
        plan_split(PATIENTS[:5], n_val=4, n_test=3, seed=0)


def test_a_corrupt_split_file_is_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        plan_split(PATIENTS, n_val=1, n_test=1, seed=0, existing={"p00": "validation"})


def test_split_files_round_trip_and_a_missing_file_means_no_assignments(tmp_path):
    assert read_splits(tmp_path / "nope.csv") == {}
    write_splits(tmp_path / "s.csv", {"b": "val", "a": "train"})
    assert read_splits(tmp_path / "s.csv") == {"a": "train", "b": "val"}


# ------------------------------------------------------- assign_source_splits
def _nhrd_manifest(n_patients=10):
    return pd.DataFrame({
        "volume_id": [f"v{i}" for i in range(n_patients)],
        "patient_id": [f"pat{i}" for i in range(n_patients)],
        "source_name": "nhrd_local",
    })


def _qc(manifest, failed=()):
    return pd.DataFrame({"volume_id": manifest.volume_id, "passed": ~manifest.volume_id.isin(failed)})


def _nhrd_cfg(n_val=2, n_test=2):
    return SourceConfig(format="dicom", manifest_builder="folder", split=SplitConfig(n_val_patients=n_val, n_test_patients=n_test))


def test_only_qc_passed_volumes_count_and_a_fully_failed_patient_is_not_assigned(tmp_path):
    m = _nhrd_manifest()
    report = assign_source_splits(m, "nhrd_local", _nhrd_cfg(), tmp_path / "s.csv", _qc(m, failed={"v0", "v1"}))
    assert "pat0" not in report.mapping and "pat1" not in report.mapping
    assert len(report.mapping) == 8 and report.n_unusable_volumes == 2
    assert report.patients_per_split == {"train": 4, "val": 2, "test": 2}


def test_the_split_is_frozen_on_disk_and_a_rerun_changes_nothing(tmp_path):
    m, path = _nhrd_manifest(), tmp_path / "s.csv"
    first = assign_source_splits(m, "nhrd_local", _nhrd_cfg(), path, _qc(m))
    again = assign_source_splits(m, "nhrd_local", _nhrd_cfg(), path, _qc(m), seed=12345)
    assert again.mapping == first.mapping and again.n_new_patients == 0 and again.n_kept_patients == 10
    assert read_splits(path) == first.mapping


def test_a_top_up_assigns_only_the_new_patients(tmp_path):
    m, path = _nhrd_manifest(8), tmp_path / "s.csv"
    first = assign_source_splits(m, "nhrd_local", _nhrd_cfg(), path, _qc(m))
    bigger = _nhrd_manifest(12)
    top_up = assign_source_splits(bigger, "nhrd_local", _nhrd_cfg(), path, _qc(bigger))
    assert top_up.n_new_patients == 4
    assert all(top_up.mapping[p] == s for p, s in first.mapping.items())
    assert all(top_up.mapping[f"pat{i}"] == "train" for i in range(8, 12))


def test_a_dry_run_writes_nothing(tmp_path):
    m, path = _nhrd_manifest(), tmp_path / "s.csv"
    assign_source_splits(m, "nhrd_local", _nhrd_cfg(), path, _qc(m), dry_run=True)
    assert not path.exists()


def test_a_missing_val_count_is_an_error(tmp_path):
    m = _nhrd_manifest()
    with pytest.raises(ValueError, match="n_val_patients"):
        assign_source_splits(m, "nhrd_local", SourceConfig(format="dicom", manifest_builder="folder"), tmp_path / "s.csv", _qc(m))


def test_ctrate_test_is_the_official_valid_pool_and_val_is_drawn_only_from_train_pool(tmp_path):
    rows = [(f"train_{i}_a_1", f"train_{i}", "train") for i in range(1, 11)]
    rows += [(f"valid_{i}_a_1", f"valid_{i}", "valid") for i in range(1, 4)]
    m = pd.DataFrame(rows, columns=["volume_id", "patient_id", "source_split"]).assign(source_name="ctrate")
    cfg = SourceConfig(manifest_builder="ctrate", split=SplitConfig(n_val_patients=3))

    report = assign_source_splits(m, "ctrate", cfg, tmp_path / "c.csv", _qc(m))

    assert all(report.mapping[f"valid_{i}"] == "test" for i in range(1, 4))  # a rule, not a draw
    val = [p for p, s in report.mapping.items() if s == "val"]
    assert len(val) == 3 and all(p.startswith("train_") for p in val)
    assert report.patients_per_split == {"train": 7, "val": 3, "test": 3}


# --------------------------------------------------------------- apply_splits
def test_apply_splits_marks_qc_failures_excluded_and_new_patients_unassigned():
    m = _nhrd_manifest(4)
    qc = _qc(m, failed={"v0"})
    out = apply_splits(m, {"nhrd_local": {"pat0": "train", "pat1": "val", "pat2": "test"}}, qc)
    assert out.set_index("volume_id").split.to_dict() == {"v0": "excluded", "v1": "val", "v2": "test", "v3": "unassigned"}
    assert out.set_index("volume_id").qc_passed.to_dict() == {"v0": False, "v1": True, "v2": True, "v3": True}


def test_apply_splits_survives_a_qc_report_round_tripped_through_csv(tmp_path):
    m = _nhrd_manifest(3)
    _qc(m, failed={"v1"}).to_csv(tmp_path / "qc.csv", index=False)
    qc = pd.read_csv(tmp_path / "qc.csv")  # booleans come back as numpy bools
    out = apply_splits(m, {"nhrd_local": {"pat0": "train", "pat1": "train", "pat2": "val"}}, qc)
    assert out.split.tolist() == ["train", "excluded", "val"]


def test_a_volume_missing_from_the_qc_report_is_excluded_not_assumed_good():
    m = _nhrd_manifest(2)
    qc = _qc(m).iloc[:1]
    out = apply_splits(m, {"nhrd_local": {"pat0": "train", "pat1": "train"}}, qc)
    assert out.split.tolist() == ["train", "excluded"]


def test_apply_splits_without_a_qc_report_does_not_exclude_anything():
    m = _nhrd_manifest(2)
    out = apply_splits(m, {"nhrd_local": {"pat0": "train", "pat1": "val"}})
    assert out.split.tolist() == ["train", "val"] and "qc_passed" not in out.columns


def test_the_same_patient_name_in_two_sources_is_two_different_patients():
    m = pd.DataFrame({"volume_id": ["a", "b"], "patient_id": ["x", "x"], "source_name": ["s1", "s2"]})
    out = apply_splits(m, {"s1": {"x": "train"}, "s2": {"x": "test"}})  # must not look like a leak
    assert out.split.tolist() == ["train", "test"]
