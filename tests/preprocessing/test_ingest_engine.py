"""The ingest loop, driven by a fake fetcher that writes real synthetic scans so the
real preprocessing core runs end to end."""
import shutil

import numpy as np
import pandas as pd
import pytest

from ct_preprocessing.config import DataConfig, PathsConfig, SourceConfig
from ct_preprocessing.ingest import state
from ct_preprocessing.ingest.base import FetchError, FetchReport
from ct_preprocessing.ingest.engine import (
    IngestOptions, ingest_status, preprocess_rows, resolve_patient_id_source, run_ingest,
)
from ct_preprocessing.manifest import build_manifest_ctrate
from ct_preprocessing.preprocess import PreprocessConfig, preprocess_one
from ct_preprocessing.quality import QCThresholds
from ct_preprocessing.runs import Run

CHUNKS = {"0000": ["train_1_a_1", "train_2_a_1"], "0001": ["train_3_a_1"], "0002": ["train_4_a_1"]}


class FakeFetcher:
    def __init__(self, nifti, chunks=None, *, corrupt=(), fail_chunks=(), break_rows=()):
        self.nifti, self.chunks = nifti, chunks or CHUNKS
        self.corrupt, self.fail_chunks, self.break_rows = set(corrupt), set(fail_chunks), set(break_rows)
        self.fetch_calls, self.downloaded, self.patient_id_sources = [], [], []

    def chunk_ids(self):
        return list(self.chunks)

    def fetch(self, chunk_id, raw_dir, *, cache_fresh):
        self.fetch_calls.append(chunk_id)
        if chunk_id in self.fail_chunks:
            raise FetchError("network is down")
        for v in self.chunks[chunk_id]:
            if cache_fresh(v):
                continue
            self.downloaded.append(v)
            dest = raw_dir / f"{v}.nii.gz"
            if v in self.corrupt:
                dest.write_bytes(b"garbage")
            else:
                shutil.copyfile(self.nifti, dest)
        return FetchReport()

    def build_rows(self, chunk_id, raw_dir, *, patient_id_source):
        self.patient_id_sources.append(patient_id_source)
        if chunk_id in self.break_rows:
            raise ZeroDivisionError("bug in a fetcher")
        return build_manifest_ctrate(self.chunks[chunk_id])


@pytest.fixture
def nifti(synthetic_nifti):
    return synthetic_nifti[0]


def make_cfg(tmp_path, builder="ctrate"):
    d = tmp_path / "data"
    paths = PathsConfig(
        raw_dir=str(d / "raw"), cache_dir=str(d / "cache"), runs_dir=str(d / "runs"),
        state_dir=str(d / "state"), splits_dir=str(d / "splits"), metadata_dir=str(d / "metadata"),
    )
    src = SourceConfig(format="nifti", manifest_builder=builder, raw_dir=str(d / "raw" / "ctrate"))
    return DataConfig(
        paths=paths, preprocess=PreprocessConfig(target_size_hw=(32, 32)), qc=QCThresholds(min_slices=10),
        sources={"ctrate": src},
    )


OPTS = dict(min_free_gb=0, workers=1)


def RUN(cfg, name="main", source="ctrate"):
    return Run(cfg.paths, source, name)


def run(cfg, fetcher, **kw):
    logs: list[str] = []
    summary = run_ingest(RUN(cfg), cfg, fetcher, IngestOptions(**{**OPTS, **kw}), log=logs.append)
    return summary, logs


def test_every_chunk_is_ingested_into_the_cache_and_the_raw_data_is_deleted(tmp_path, nifti):
    cfg, fetcher = make_cfg(tmp_path), FakeFetcher(nifti)
    summary, logs = run(cfg, fetcher)

    assert (summary.chunks_done, summary.chunks_failed, summary.volumes_ok, summary.pending_after) == (3, 0, 4, 0)
    cache = tmp_path / "data" / "cache"
    for v in ("train_1_a_1", "train_2_a_1", "train_3_a_1", "train_4_a_1"):
        arr = np.load(cache / f"{v}.npy")
        assert arr.dtype == np.int16 and arr.shape[1:] == (32, 32)  # raw HU, not windowed 3-channel float
        assert (cache / f"{v}.meta.json").exists()
    scratch = tmp_path / "data" / "raw" / "ctrate"
    assert [p.name for p in scratch.iterdir()] == [state.SCRATCH_SENTINEL]  # raw never accumulates
    assert set(state.done_chunks(RUN(cfg))) == set(CHUNKS)
    assert any("chunk 0000" in line for line in logs)


def test_chunk_manifests_carry_the_stats_and_no_split(tmp_path, nifti):
    cfg = make_cfg(tmp_path)
    run(cfg, FakeFetcher(nifti))
    rows = pd.read_csv(RUN(cfg).chunk_manifest_path("0000"), dtype={"ingest_chunk": str})
    assert rows.volume_id.tolist() == ["train_1_a_1", "train_2_a_1"]
    assert {"n_slices", "npy_path", "spacing_z_mm", "crop_shape", "qc_passed", "source_name", "ingest_chunk"} <= set(rows.columns)
    assert "split" not in rows.columns
    assert rows.n_slices.notna().all() and set(rows.source_name) == {"ctrate"} and set(rows.ingest_chunk) == {"0000"}


def test_running_again_does_nothing_because_finished_chunks_are_skipped(tmp_path, nifti):
    cfg, fetcher = make_cfg(tmp_path), FakeFetcher(nifti)
    run(cfg, fetcher)
    calls = len(fetcher.fetch_calls)
    summary, _ = run(cfg, fetcher)
    assert len(fetcher.fetch_calls) == calls and summary.chunks_done == 0


def test_max_chunks_stops_early_and_the_next_run_continues(tmp_path, nifti):
    cfg, fetcher = make_cfg(tmp_path), FakeFetcher(nifti)
    summary, _ = run(cfg, fetcher, max_chunks=1)
    assert summary.chunks_done == 1 and summary.pending_after == 2 and "max-chunks" in summary.stopped
    summary, _ = run(cfg, fetcher)
    assert summary.chunks_done == 2 and summary.pending_after == 0


def test_a_half_finished_chunk_resumes_without_refetching_what_is_already_cached(tmp_path, nifti):
    cfg = make_cfg(tmp_path)
    run(cfg, FakeFetcher(nifti), only_chunk="0000")
    state.clear_done(RUN(cfg), "0000")  # simulate a crash right before the marker was written

    fetcher = FakeFetcher(nifti)
    run(cfg, fetcher, only_chunk="0000")

    assert fetcher.downloaded == []  # both volumes were already cached
    rows = pd.read_csv(RUN(cfg).chunk_manifest_path("0000"))
    assert rows.n_slices.notna().all()  # the stats were rebuilt from the cache sidecars, not lost


def test_the_run_stops_cleanly_when_free_disk_is_below_the_limit(tmp_path, nifti):
    cfg, fetcher = make_cfg(tmp_path), FakeFetcher(nifti)
    summary, _ = run(cfg, fetcher, min_free_gb=1e9)
    assert "min_free_gb" in summary.stopped and summary.chunks_done == 0 and fetcher.fetch_calls == []


def test_a_corrupt_volume_is_recorded_without_failing_the_chunk_and_can_be_retried(tmp_path, nifti):
    cfg = make_cfg(tmp_path)
    summary, _ = run(cfg, FakeFetcher(nifti, corrupt={"train_2_a_1"}))
    assert (summary.chunks_done, summary.volumes_ok, summary.volumes_failed) == (3, 3, 1)
    info = state.done_chunks(RUN(cfg))["0000"]
    assert info["n_failed"] == 1 and list(info["failed"]) == ["train_2_a_1"]
    assert not (tmp_path / "data" / "cache" / "train_2_a_1.npy").exists()

    fixed = FakeFetcher(nifti)  # the file is fine this time
    run(cfg, fixed, retry_failed=True)
    assert fixed.downloaded == ["train_2_a_1"]  # only the failed volume is fetched again
    assert state.done_chunks(RUN(cfg))["0000"]["n_failed"] == 0
    assert (tmp_path / "data" / "cache" / "train_2_a_1.npy").exists()


def test_a_failed_chunk_gets_a_failed_marker_and_is_retried_on_the_next_run(tmp_path, nifti):
    cfg = make_cfg(tmp_path)
    summary, _ = run(cfg, FakeFetcher(nifti, fail_chunks={"0001"}))
    assert summary.chunks_done == 2 and summary.chunks_failed == 1
    assert state.failed_chunks(RUN(cfg))["0001"]["attempts"] == 1

    run(cfg, FakeFetcher(nifti, fail_chunks={"0001"}))
    assert state.failed_chunks(RUN(cfg))["0001"]["attempts"] == 2

    summary, _ = run(cfg, FakeFetcher(nifti))
    assert summary.chunks_done == 1 and state.failed_chunks(RUN(cfg)) == {}


def test_repeated_failures_stop_the_run_instead_of_grinding_through_every_chunk(tmp_path, nifti):
    cfg = make_cfg(tmp_path)
    many = {f"{i:04d}": ["train_1_a_1"] for i in range(6)}
    fetcher = FakeFetcher(nifti, many, fail_chunks=set(many))
    summary, _ = run(cfg, fetcher, max_consecutive_failures=2)
    assert len(fetcher.fetch_calls) == 2 and "in a row" in summary.stopped


def test_an_unexpected_bug_in_one_chunk_does_not_end_an_unattended_run(tmp_path, nifti):
    cfg = make_cfg(tmp_path)
    summary, logs = run(cfg, FakeFetcher(nifti, break_rows={"0001"}))
    assert summary.chunks_done == 2 and summary.chunks_failed == 1
    assert "ZeroDivisionError" in state.failed_chunks(RUN(cfg))["0001"]["error"]


def test_a_scratch_folder_with_foreign_files_is_refused_and_nothing_is_deleted(tmp_path, nifti):
    cfg = make_cfg(tmp_path)
    scratch = tmp_path / "data" / "raw" / "ctrate"
    scratch.mkdir(parents=True)
    (scratch / "precious.txt").write_text("irreplaceable")
    with pytest.raises(state.ScratchError, match="not created by ingest"):
        run(cfg, FakeFetcher(nifti))
    assert (scratch / "precious.txt").read_text() == "irreplaceable"


def test_leftovers_from_a_crashed_run_in_our_own_scratch_are_cleared(tmp_path, nifti):
    cfg = make_cfg(tmp_path)
    run(cfg, FakeFetcher(nifti), max_chunks=0)  # just prepares the scratch folder
    scratch = tmp_path / "data" / "raw" / "ctrate"
    (scratch / "half_downloaded.nii.gz").write_bytes(b"x")
    run(cfg, FakeFetcher(nifti), only_chunk="0000")
    assert [p.name for p in scratch.iterdir()] == [state.SCRATCH_SENTINEL]


def test_a_dry_run_changes_nothing(tmp_path, nifti):
    cfg, fetcher = make_cfg(tmp_path), FakeFetcher(nifti)
    summary, logs = run(cfg, fetcher, dry_run=True)
    assert fetcher.fetch_calls == [] and summary.pending_after == 3
    assert not (tmp_path / "data").exists() and any("would ingest chunk 0000" in line for line in logs)


def test_an_unknown_chunk_is_rejected(tmp_path, nifti):
    with pytest.raises(ValueError, match="not one of"):
        run(make_cfg(tmp_path), FakeFetcher(nifti), only_chunk="9999")


def test_the_status_report_shows_progress_and_the_measured_cache_size(tmp_path, nifti):
    cfg, fetcher = make_cfg(tmp_path), FakeFetcher(nifti)
    run(cfg, fetcher, max_chunks=2)
    text = ingest_status(RUN(cfg), cfg, fetcher)
    assert "chunks done: 2 / 3" in text and "volumes ok: 3" in text and "MB/volume" in text


# ------------------------------------------------------------ preprocess_rows
def _rows(ids):
    return build_manifest_ctrate(ids)


def test_a_cached_volume_counts_as_done_even_after_its_raw_file_was_deleted(tmp_path, nifti):
    cache, raw = tmp_path / "cache", tmp_path / "raw"
    raw.mkdir()
    cfg = PreprocessConfig(target_size_hw=(32, 32))
    preprocess_one(nifti, cache, cfg, volume_id="train_1_a_1")
    result = preprocess_rows(_rows(["train_1_a_1"]), raw_dir=raw, cache_dir=cache, preprocess_cfg=cfg, qc=QCThresholds(min_slices=10))
    assert (result.n_cached, result.n_ok, result.failed) == (1, 0, {})
    assert result.stats.loc[0, "n_slices"] > 0


def test_a_volume_with_no_raw_file_and_no_cache_is_a_recorded_failure_not_a_crash(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    result = preprocess_rows(
        _rows(["train_9_a_1"]), raw_dir=raw, cache_dir=tmp_path / "cache",
        preprocess_cfg=PreprocessConfig(target_size_hw=(32, 32)), qc=QCThresholds(min_slices=10),
    )
    assert "raw file missing" in result.failed["train_9_a_1"] and result.n_ok == 0


def test_a_changed_config_makes_the_cache_stale_and_needs_the_raw_file_again(tmp_path, nifti):
    cache, raw = tmp_path / "cache", tmp_path / "raw"
    raw.mkdir()
    preprocess_one(nifti, cache, PreprocessConfig(target_size_hw=(32, 32)), volume_id="train_1_a_1")
    result = preprocess_rows(
        _rows(["train_1_a_1"]), raw_dir=raw, cache_dir=cache,
        preprocess_cfg=PreprocessConfig(target_size_hw=(16, 16)), qc=QCThresholds(min_slices=10),
    )
    assert "train_1_a_1" in result.failed  # stale, and the raw scan is gone: it has to be re-ingested


def test_force_reprocesses_even_a_fresh_cache(tmp_path, nifti):
    cache, raw = tmp_path / "cache", tmp_path / "raw"
    raw.mkdir()
    shutil.copyfile(nifti, raw / "train_1_a_1.nii.gz")
    cfg = PreprocessConfig(target_size_hw=(32, 32))
    preprocess_one(nifti, cache, cfg, volume_id="train_1_a_1")
    result = preprocess_rows(_rows(["train_1_a_1"]), raw_dir=raw, cache_dir=cache, preprocess_cfg=cfg,
                             qc=QCThresholds(min_slices=10), force=True)
    assert (result.n_ok, result.n_cached) == (1, 0)


def test_several_workers_give_the_same_result_as_one(tmp_path, nifti):
    cache, raw = tmp_path / "cache", tmp_path / "raw"
    raw.mkdir()
    ids = ["train_1_a_1", "train_2_a_1", "train_3_a_1"]
    for v in ids:
        shutil.copyfile(nifti, raw / f"{v}.nii.gz")
    result = preprocess_rows(_rows(ids), raw_dir=raw, cache_dir=cache, preprocess_cfg=PreprocessConfig(target_size_hw=(32, 32)),
                             qc=QCThresholds(min_slices=10), workers=2)
    assert result.n_ok == 3 and sorted(result.stats.volume_id) == sorted(ids)


# --------------------------------------------------- the sticky "auto" decision
class FolderFetcher(FakeFetcher):
    """A folder-type source (NHRD-like) that reports the patient_id_source it resolved."""

    def fetch(self, chunk_id, raw_dir, *, cache_fresh):
        for v in self.chunks[chunk_id]:
            (raw_dir / v).mkdir()
            shutil.copyfile(self.nifti, raw_dir / v / "scan.nii.gz")
        return FetchReport()

    def build_rows(self, chunk_id, raw_dir, *, patient_id_source):
        self.patient_id_sources.append(patient_id_source)
        rows = pd.DataFrame({
            "volume_id": self.chunks[chunk_id], "patient_id": self.chunks[chunk_id],
            "scan_path": [f"{v}/scan.nii.gz" for v in self.chunks[chunk_id]], "format": "nifti",
        })
        rows.attrs["patient_id_source"] = "path"
        return rows


def test_auto_patient_grouping_is_decided_once_and_every_later_chunk_follows_it(tmp_path, nifti):
    cfg = make_cfg(tmp_path, builder="folder")
    fetcher = FolderFetcher(nifti, {"a": ["p1"], "b": ["p2"], "c": ["p3"]})
    run(cfg, fetcher)
    assert fetcher.patient_id_sources == ["auto", "path", "path"]
    assert state.read_patient_id_source(cfg.paths, "ctrate") == "path"


def test_an_explicit_patient_id_source_always_beats_the_remembered_one(tmp_path):
    paths = make_cfg(tmp_path).paths
    state.write_patient_id_source(paths, "ctrate", "path")
    assert resolve_patient_id_source(paths, "ctrate", "dicom_tag") == "dicom_tag"
    assert resolve_patient_id_source(paths, "ctrate", "auto") == "path"


# ----------------------------------------------------------------- one run at a time
def test_a_second_ingest_on_the_same_source_is_refused_and_touches_nothing(tmp_path, nifti):
    from ct_preprocessing.ingest.lock import AlreadyRunning, SourceLock

    cfg = make_cfg(tmp_path)
    with SourceLock(state.source_state_dir(cfg.paths, "ctrate") / "ingest.lock", "ctrate"):
        with pytest.raises(AlreadyRunning, match="already running"):
            run(cfg, FakeFetcher(nifti))
    assert not (tmp_path / "data" / "cache").exists()  # the refused run did nothing
    summary, _ = run(cfg, FakeFetcher(nifti))  # and once the first run is over, it works
    assert summary.chunks_done == 3


def test_the_lock_is_released_when_a_run_ends_so_there_is_never_a_stale_lock(tmp_path, nifti):
    cfg = make_cfg(tmp_path)
    run(cfg, FakeFetcher(nifti), max_chunks=1)
    run(cfg, FakeFetcher(nifti), max_chunks=1)  # would raise if the first had left a lock behind


def test_different_sources_do_not_block_each_other(tmp_path):
    from ct_preprocessing.ingest.lock import SourceLock

    paths = make_cfg(tmp_path).paths
    with SourceLock(state.source_state_dir(paths, "ctrate") / "ingest.lock", "ctrate"):
        with SourceLock(state.source_state_dir(paths, "nhrd_local") / "ingest.lock", "nhrd_local"):
            pass


def test_a_dry_run_with_retry_failed_does_not_clear_any_marker(tmp_path, nifti):
    cfg = make_cfg(tmp_path)
    run(cfg, FakeFetcher(nifti, corrupt={"train_2_a_1"}))
    summary, logs = run(cfg, FakeFetcher(nifti), dry_run=True, retry_failed=True)
    assert "0000" in state.done_chunks(RUN(cfg))  # still marked done
    assert summary.pending_after == 1 and any("would ingest chunk 0000" in line for line in logs)
