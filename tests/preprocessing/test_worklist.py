"""The CT-RATE worklist: which volumes, in what order, grouped how."""
import pandas as pd
import pytest

from ct_preprocessing.ingest.worklist import (
    WorklistConflict, build_worklist, format_summary, read_worklist, write_worklist,
)


def _meta(pool: str, patients: range, scans="a", recons=(1, 2), **extra) -> pd.DataFrame:
    rows = [
        {"VolumeName": f"{pool}_{p}_{s}_{r}.nii.gz", **extra}
        for p in patients for s in scans for r in recons
    ]
    return pd.DataFrame(rows)


TRAIN = _meta("train", range(1, 11), scans="ab")   # 10 patients x 2 scans x 2 recons = 40 volumes
VALID = _meta("valid", range(1, 4))                # 3 patients x 1 scan x 2 recons = 6 volumes


def _build(**kw):
    kw.setdefault("chunk_size", 4)
    return build_worklist(TRAIN, VALID, **kw)


def test_the_test_pool_is_taken_whole_and_train_one_reconstruction_per_scan():
    wl, summary = _build()
    test, train = wl[wl.source_split == "valid"], wl[wl.source_split == "train"]
    assert len(test) == 6 and sorted(test.reconstruction_id.unique()) == [1, 2]
    assert len(train) == 20 and set(train.reconstruction_id) == {1}  # 10 patients x 2 scans
    assert summary["test_volumes"] == 6 and summary["train_volumes"] == 20


def test_train_pool_all_keeps_both_reconstructions():
    wl, _ = _build(train_pool="all")
    assert len(wl[wl.source_split == "train"]) == 40


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


def test_a_cap_is_prefix_stable_so_finished_chunks_never_change():
    small, _ = _build(max_train_patients=4)
    large, _ = _build(max_train_patients=8)
    small_train = small[small.source_split == "train"].volume_id.tolist()
    large_train = large[large.source_split == "train"].volume_id.tolist()
    assert large_train[: len(small_train)] == small_train
    # and the chunk each shared volume belongs to is identical too
    shared = small[small.volume_id.isin(small_train)].set_index("volume_id").chunk
    assert (large.set_index("volume_id").chunk.loc[shared.index] == shared).all()


def test_the_cap_counts_patients_not_volumes():
    wl, summary = _build(max_train_patients=3)
    assert wl[wl.source_split == "train"].patient_id.nunique() == 3
    assert summary["train_patients"] == 3 and summary["train_patients_available"] == 10


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
    wl, summary = build_worklist(train, valid, chunk_size=4, max_combined_gb=1.5)
    assert "train_3_a_1" not in set(wl.volume_id) and summary["screened_out"] == 1
    assert "valid_1_a_1" in set(wl.volume_id)


def test_rows_from_the_wrong_pool_or_with_bad_ids_are_ignored_and_counted():
    train = pd.concat([TRAIN, pd.DataFrame({"VolumeName": ["valid_99_a_1.nii.gz", "garbage.nii.gz"]})])
    wl, summary = build_worklist(train, VALID, chunk_size=4)
    assert "valid_99_a_1" not in set(wl.volume_id)
    assert summary["skipped_wrong_pool"] == 1 and summary["skipped_unparsable"] == 1


def test_write_worklist_refuses_to_overwrite_unless_asked(tmp_path):
    wl, _ = _build()
    path = tmp_path / "wl.csv"
    write_worklist(wl, path)
    with pytest.raises(FileExistsError):
        write_worklist(wl, path)
    write_worklist(wl, path, overwrite=True)  # nothing ingested yet -> fine
    assert read_worklist(path).equals(wl)


def test_rebuilding_may_not_change_a_chunk_that_was_already_ingested(tmp_path):
    path = tmp_path / "wl.csv"
    write_worklist(_build(chunk_size=4)[0], path)
    # a different chunk size reshuffles which volumes sit in chunk 0000 -> its done marker would lie
    with pytest.raises(WorklistConflict):
        write_worklist(_build(chunk_size=5)[0], path, overwrite=True, done_chunks={"0000"})
    # but a larger cap only appends, so the same done chunks stay valid
    write_worklist(_build(max_train_patients=8)[0], path, overwrite=True, done_chunks=set())
    write_worklist(_build(max_train_patients=10)[0], path, overwrite=True, done_chunks={"0000", "0001", "0002"})


def test_the_summary_projects_the_cache_size():
    _, summary = _build()
    text = format_summary(summary, est_mb_per_volume=22)
    assert "26 volumes" in text and "projected cache" in text


def test_the_test_pool_can_be_capped_for_a_pilot_and_is_prefix_stable_too():
    small, summary = _build(max_test_patients=1)
    large, _ = _build(max_test_patients=2)
    test_small = small[small.source_split == "valid"].volume_id.tolist()
    test_large = large[large.source_split == "valid"].volume_id.tolist()
    assert small[small.source_split == "valid"].patient_id.nunique() == 1 and summary["test_patients_available"] == 3
    assert test_large[: len(test_small)] == test_small
    assert _build()[0][lambda d: d.source_split == "valid"].patient_id.nunique() == 3  # the default keeps it whole
