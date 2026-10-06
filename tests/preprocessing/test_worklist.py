"""The CT-RATE worklist: which volumes, in what order, grouped how -- and which kernel."""
import pandas as pd
import pytest

from ct_preprocessing.ingest.worklist import (
    build_survey_worklist, build_worklist, format_summary, patients_with_kernel, read_worklist,
    select_child_patients, write_worklist,
)

# Philips: reconstruction 1 is the sharp lung series (kernel YA), reconstruction 2 the soft one (B)
TABLE = {("philips", "ya"): "sharp", ("philips", "b"): "soft"}
KERNEL = {1: "YA", 2: "B"}


def _meta(pool, patients, scans="a", recons=(1, 2), **extra):
    rows = [
        {"VolumeName": f"{pool}_{p}_{s}_{r}.nii.gz", "Manufacturer": "Philips", "ConvolutionKernel": KERNEL[r], **extra}
        for p in patients for s in scans for r in recons
    ]
    return pd.DataFrame(rows)


TRAIN = _meta("train", range(1, 11), scans="ab")   # 10 patients x 2 scans x 2 recons = 40 volumes
VALID = _meta("valid", range(1, 4))                # 3 patients x 1 scan x 2 recons = 6 volumes


def _build(train=TRAIN, valid=VALID, **kw):
    kw.setdefault("chunk_size", 4)
    kw.setdefault("kernel_table", TABLE)
    return build_worklist(train, valid, **kw)


# ------------------------------------------------------------------- what is taken
def test_the_test_pool_is_taken_whole_with_both_kernels_and_train_one_sharp_reconstruction_per_scan():
    wl, summary = _build()
    test, train = wl[wl.source_split == "valid"], wl[wl.source_split == "train"]
    assert len(test) == 6 and set(test.kernel_class) == {"sharp", "soft"}
    assert len(train) == 20 and set(train.reconstruction_id) == {1} and set(train.kernel_class) == {"sharp"}
    assert summary["test_volumes"] == 6 and summary["train_volumes"] == 20
    assert summary["test_kernel_classes"] == {"sharp": 3, "soft": 3}


def test_a_soft_run_takes_the_soft_reconstruction_and_the_test_pool_is_unchanged():
    sharp, _ = _build(train_kernel="sharp")
    soft, _ = _build(train_kernel="soft")
    train = soft[soft.source_split == "train"]
    assert set(train.reconstruction_id) == {2} and set(train.kernel) == {"B"} and len(train) == 20
    assert soft[soft.source_split == "valid"].volume_id.tolist() == sharp[sharp.source_split == "valid"].volume_id.tolist()


def test_the_worklist_names_each_volumes_kernel_and_class():
    wl, _ = _build()
    row = wl[wl.volume_id == "valid_1_a_2"].iloc[0]
    assert (row.manufacturer, row.kernel, row.kernel_class) == ("Philips", "B", "soft")
    assert {"kernel", "kernel_class", "manufacturer"} <= set(wl.columns)


def test_scans_without_the_wanted_kernel_are_skipped_and_counted():
    only_soft = _meta("train", [11], recons=(2,))  # patient 11's only scan has just the soft reconstruction
    train = pd.concat([TRAIN, only_soft])
    sharp, summary = _build(train=train)
    assert "train_11_a_2" not in set(sharp.volume_id) and "train_11" not in set(sharp.patient_id)
    assert summary["train_scans_skipped_no_kernel"] == 1 and summary["train_patients_without_kernel"] == 1
    soft, _ = _build(train=train, train_kernel="soft")
    assert "train_11_a_2" in set(soft.volume_id)


def test_a_kernel_that_is_not_in_the_table_is_never_chosen():
    train = pd.concat([TRAIN, _meta("train", [11], recons=(1,)).assign(ConvolutionKernel="MYSTERY")])
    wl, _ = _build(train=train)
    assert "train_11" not in set(wl.patient_id)


def test_the_metadata_must_carry_manufacturer_and_kernel():
    with pytest.raises(ValueError, match="no column"):
        build_worklist(TRAIN.drop(columns="ConvolutionKernel"), VALID, kernel_table=TABLE, chunk_size=4)
    with pytest.raises(ValueError, match="'sharp' or 'soft'"):
        _build(train_kernel="medium")


# ------------------------------------------------------------------- order and chunks
def test_test_pool_chunks_come_first_and_no_chunk_mixes_the_pools():
    wl, summary = _build()
    assert summary["test_chunks"] == 2  # 6 volumes / chunk_size 4
    first_train_chunk = wl[wl.source_split == "train"].chunk.min()
    assert first_train_chunk == summary["test_chunks"]
    for _, group in wl.groupby("chunk"):
        assert group.source_split.nunique() == 1
        assert len(group) <= 4
    assert list(wl.order) == list(range(len(wl)))


def test_a_patients_volumes_stay_together_in_the_seeded_order():
    wl, _ = _build()
    train = wl[wl.source_split == "train"]
    seen = list(dict.fromkeys(train.patient_id))
    assert train.patient_id.tolist() == [p for p in seen for _ in range((train.patient_id == p).sum())]


def test_the_worklist_is_deterministic_for_a_seed_and_differs_across_seeds():
    a, _ = _build(seed=0)
    b, _ = _build(seed=0)
    c, _ = _build(seed=1)
    assert a.equals(b)
    assert a.volume_id.tolist() != c.volume_id.tolist()


def test_a_sharp_run_and_a_soft_run_walk_the_same_patient_order():
    """The shuffle is over ALL train patients, not only those with the wanted kernel, so a cap picks the
    same first patients whichever kernel is asked for -- the soft run of 4 patients is the sharp run's 4."""
    sharp, _ = _build(max_train_patients=4, train_kernel="sharp")
    soft, _ = _build(max_train_patients=4, train_kernel="soft")
    order = lambda d: list(dict.fromkeys(d[d.source_split == "train"].patient_id))  # noqa: E731
    assert order(sharp) == order(soft)


def test_a_cap_is_prefix_stable_so_finished_chunks_never_change():
    small, _ = _build(max_train_patients=4)
    large, _ = _build(max_train_patients=8)
    small_train = small[small.source_split == "train"].volume_id.tolist()
    large_train = large[large.source_split == "train"].volume_id.tolist()
    assert large_train[: len(small_train)] == small_train
    shared = small[small.volume_id.isin(small_train)].set_index("volume_id").chunk
    assert (large.set_index("volume_id").chunk.loc[shared.index] == shared).all()


def test_the_cap_counts_patients_that_have_the_kernel_not_volumes():
    wl, summary = _build(max_train_patients=3)
    assert wl[wl.source_split == "train"].patient_id.nunique() == 3
    assert summary["train_patients"] == 3 and summary["train_patients_available"] == 10
    # a patient with no sharp scan does not use up a place in the cap
    train = pd.concat([TRAIN, _meta("train", [11], recons=(2,))])
    capped, _ = _build(train=train, max_train_patients=10)
    assert capped[capped.source_split == "train"].patient_id.nunique() == 10


def test_the_test_pool_can_be_capped_for_a_pilot_and_is_prefix_stable_too():
    small, summary = _build(max_test_patients=1)
    large, _ = _build(max_test_patients=2)
    test_small = small[small.source_split == "valid"].volume_id.tolist()
    test_large = large[large.source_split == "valid"].volume_id.tolist()
    assert small[small.source_split == "valid"].patient_id.nunique() == 1 and summary["test_patients_available"] == 3
    assert test_large[: len(test_small)] == test_small
    assert _build()[0][lambda d: d.source_split == "valid"].patient_id.nunique() == 3  # the default keeps it whole


# ------------------------------------------------------------------- other behaviour kept
def test_repo_paths_follow_the_confirmed_fixed_layout():
    wl, _ = _build()
    row = wl[wl.volume_id == "valid_1_a_2"].iloc[0]
    assert row.repo_path == "dataset/valid_fixed/valid_1/valid_1_a/valid_1_a_2.nii.gz"
    row = wl[wl.source_split == "train"].iloc[0]
    assert row.repo_path.startswith(f"dataset/train_fixed/{row.patient_id}/")


def test_the_memory_screen_drops_big_train_volumes_but_never_test_volumes():
    big = dict(Rows=1024, Columns=1024, NumberofSlices=900, XYSpacing="[0.5, 0.5]", ZSpacing=0.5)
    small = dict(Rows=512, Columns=512, NumberofSlices=100, XYSpacing="[0.9, 0.9]", ZSpacing=1.5)
    train = pd.concat([_meta("train", range(1, 3), recons=(1,), **small), _meta("train", range(3, 4), recons=(1,), **big)])
    valid = _meta("valid", range(1, 2), recons=(1,), **big)  # a huge test volume must stay
    wl, summary = _build(train=train, valid=valid, max_combined_gb=1.5)
    assert "train_3_a_1" not in set(wl.volume_id) and summary["screened_out"] == 1
    assert "valid_1_a_1" in set(wl.volume_id)


def test_rows_from_the_wrong_pool_or_with_bad_ids_are_ignored_and_counted():
    junk = pd.DataFrame({"VolumeName": ["valid_99_a_1.nii.gz", "garbage.nii.gz"], "Manufacturer": "Philips", "ConvolutionKernel": "YA"})
    wl, summary = _build(train=pd.concat([TRAIN, junk]))
    assert "valid_99_a_1" not in set(wl.volume_id)
    assert summary["skipped_wrong_pool"] == 1 and summary["skipped_unparsable"] == 1


def test_a_duplicated_metadata_row_does_not_list_a_volume_twice():
    wl, _ = _build(train=pd.concat([TRAIN, TRAIN.iloc[:2]]), valid=pd.concat([VALID, VALID.iloc[:2]]))
    assert not wl.volume_id.duplicated().any()


def test_write_worklist_never_overwrites(tmp_path):
    wl, _ = _build()
    path = tmp_path / "run" / "worklist.csv"
    write_worklist(wl, path)
    with pytest.raises(FileExistsError, match="never overwritten"):
        write_worklist(wl, path)
    assert read_worklist(path).equals(wl)


def test_the_summary_projects_the_cache_size_and_reports_the_kernel():
    _, summary = _build()
    text = format_summary(summary, est_mb_per_volume=22)
    assert "26 volumes" in text and "projected cache" in text
    assert "sharp only" in text and "sharp=3, soft=3" in text and "20 of 20 train scans" in text


# ------------------------------------------------------------------- runs built from a finished run
SPLITS = {**{f"train_{p}": "train" for p in range(1, 8)}, **{f"train_{p}": "val" for p in range(8, 11)}}


def test_a_child_run_takes_the_asked_number_of_patients_from_each_split_of_the_parent():
    chosen = select_child_patients(SPLITS, set(SPLITS), n_train=3, n_val=2, seed=0)
    assert len(chosen) == 5
    assert sum(SPLITS[p] == "train" for p in chosen) == 3 and sum(SPLITS[p] == "val" for p in chosen) == 2


def test_a_child_selection_is_repeatable_and_depends_on_the_seed():
    a = select_child_patients(SPLITS, set(SPLITS), n_train=3, n_val=2, seed=0)
    assert a == select_child_patients(SPLITS, set(SPLITS), n_train=3, n_val=2, seed=0)
    assert set(a) != set(select_child_patients(SPLITS, set(SPLITS), n_train=3, n_val=2, seed=5)) or a != select_child_patients(SPLITS, set(SPLITS), n_train=3, n_val=2, seed=5)


def test_a_child_selection_only_uses_patients_that_have_the_kernel_and_says_when_there_are_too_few():
    available = {p for p in SPLITS if p not in ("train_1", "train_2")}  # two train patients have no soft scan
    chosen = select_child_patients(SPLITS, available, n_train=5, n_val=3, seed=0)
    assert not {"train_1", "train_2"} & set(chosen)
    with pytest.raises(ValueError, match="asked for 6 train patients but only 5"):
        select_child_patients(SPLITS, available, n_train=6, n_val=1, seed=0)


def test_a_child_worklist_contains_exactly_the_chosen_patients_in_the_chosen_order():
    chosen = select_child_patients(SPLITS, patients_with_kernel(TRAIN, TABLE, "soft"), n_train=3, n_val=2, seed=0)
    wl, summary = _build(train_kernel="soft", train_patients=chosen)
    train = wl[wl.source_split == "train"]
    assert list(dict.fromkeys(train.patient_id)) == chosen
    assert set(train.kernel_class) == {"soft"} and summary["train_patients"] == 5
    assert len(wl[wl.source_split == "valid"]) == 6  # the test pool is still taken whole


def test_a_child_run_cannot_pick_up_a_patient_the_parent_never_split():
    chosen = select_child_patients(SPLITS, set(SPLITS), n_train=2, n_val=1, seed=0)
    wl, _ = _build(train_kernel="soft", train_patients=[*chosen, "train_999"])  # unknown patient: no rows, so not added
    assert "train_999" not in set(wl.patient_id)


# ------------------------------------------------------------------- the kernel survey
def test_the_survey_takes_both_reconstructions_of_a_few_scans_per_kernel_pair():
    wl, summary = build_survey_worklist(TRAIN, kernel_table=TABLE, pairs_per_kernel=3, chunk_size=4)
    scans = wl.groupby(["patient_id", "scan_id"]).size()
    assert (scans == 2).all()  # always the pair
    assert summary["survey_scans"] == len(scans) == 3  # YA and B appear in the same scans, so 3 scans cover both
    assert summary["kernel_pairs"] == 2 and summary["total_volumes"] == 6
    assert set(wl.kernel_class) == {"sharp", "soft"} and "valid" not in set(wl.source_split)
    assert "kernel survey" in format_summary(summary, 22)


def test_the_survey_ignores_scans_with_a_single_reconstruction():
    train = pd.concat([TRAIN, _meta("train", [11], recons=(1,))])
    wl, _ = build_survey_worklist(train, kernel_table=TABLE, pairs_per_kernel=50, chunk_size=4)
    assert "train_11" not in set(wl.patient_id)
