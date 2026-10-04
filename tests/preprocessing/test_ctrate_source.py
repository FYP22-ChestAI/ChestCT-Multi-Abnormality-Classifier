"""CT-RATE as an ingest source: paths, metadata, memory screen, and the fetcher."""
from pathlib import Path

import pandas as pd
import pytest

from ct_preprocessing.config import PathsConfig, SourceConfig
from ct_preprocessing.ingest import ctrate
from ct_preprocessing.ingest.base import FetchError
from ct_preprocessing.ingest.ctrate import (
    CtrateFetcher, download_metadata, download_volume, load_metadata, parse_number,
    passes_memory_screen, ctrate_rel_path,
)


def test_ctrate_rel_path_matches_the_confirmed_fixed_layout():
    assert ctrate_rel_path("train_10_a_1", "train") == "dataset/train_fixed/train_10/train_10_a/train_10_a_1.nii.gz"
    assert ctrate_rel_path("valid_1002_a_2", "valid") == "dataset/valid_fixed/valid_1002/valid_1002_a/valid_1002_a_2.nii.gz"


def test_ctrate_rel_path_rejects_the_wrong_pool():
    with pytest.raises(ValueError):
        ctrate_rel_path("valid_1_a_1", "train")


def test_parse_number_takes_the_first_number_of_a_list_like_string():
    assert parse_number("[0.91796875, 0.91796875]") == pytest.approx(0.91796875)
    assert parse_number("1.5") == 1.5


def _meta(**row):
    return pd.DataFrame([{"VolumeName": "v", **row}]).set_index("VolumeName")


def test_memory_screen_lets_small_volumes_through():
    m = _meta(Rows=512, Columns=512, NumberofSlices=200, XYSpacing="[1.0, 1.0]", ZSpacing=1.5)
    assert passes_memory_screen("v", m, 10.0, (1.5, 0.75, 0.75))


def test_memory_screen_rejects_the_real_volume_that_once_ran_out_of_memory():
    m = _meta(Rows=1024, Columns=1024, NumberofSlices=237, XYSpacing="[1.0, 1.0]", ZSpacing=1.0)
    assert not passes_memory_screen("v", m, 1.5, (1.5, 0.75, 0.75))


def test_memory_screen_lets_unknown_volumes_through():
    m = _meta(Rows=1, Columns=1)
    assert passes_memory_screen("not_in_metadata", m, 1.0, (1.5, 0.75, 0.75))


# ------------------------------------------------------------------ metadata CSVs
def fake_hf(content_by_file: dict[str, bytes], root: Path, log: list | None = None):
    """A stand-in for huggingface_hub.hf_hub_download: writes the file into the
    given cache_dir (as the real one does) and returns its path."""
    def download(repo_id, filename, repo_type=None, cache_dir=None, **kw):
        if log is not None:
            log.append(filename)
        base = Path(cache_dir) if cache_dir else root / "hf_default_cache"
        target = base / "snapshots" / "abc" / Path(filename).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content_by_file[filename])
        return str(target)
    return download


def test_download_metadata_fetches_both_csvs_once_and_skips_ones_already_there(tmp_path):
    paths = PathsConfig(metadata_dir=str(tmp_path / "meta"))
    files = {ctrate.METADATA_FILES["train"]: b"VolumeName\ntrain_1_a_1.nii.gz\n",
             ctrate.METADATA_FILES["valid"]: b"VolumeName\nvalid_1_a_1.nii.gz\n"}
    log: list = []
    out = download_metadata(paths, "o/r", fake_hf(files, tmp_path, log))
    assert sorted(p.name for p in out.values()) == ["train_metadata.csv", "validation_metadata.csv"]
    assert len(log) == 2
    download_metadata(paths, "o/r", fake_hf(files, tmp_path, log))
    assert len(log) == 2  # nothing downloaded again
    download_metadata(paths, "o/r", fake_hf(files, tmp_path, log), refresh=True)
    assert len(log) == 4


def test_load_metadata_strips_the_extension_and_explains_a_missing_file(tmp_path):
    paths = PathsConfig(metadata_dir=str(tmp_path / "meta"))
    with pytest.raises(FileNotFoundError, match="make_worklist.py"):
        load_metadata(paths)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "train_metadata.csv").write_text("VolumeName,Rows\ntrain_1_a_1.nii.gz,512\n")
    (tmp_path / "meta" / "validation_metadata.csv").write_text("VolumeName,Rows\nvalid_1_a_1.nii.gz,512\n")
    meta = load_metadata(paths)
    assert meta["train"].VolumeName.tolist() == ["train_1_a_1"] and meta["valid"].VolumeName.tolist() == ["valid_1_a_1"]


# ---------------------------------------------------------------------- fetching
def _worklist():
    return pd.DataFrame({
        "chunk": [0, 0, 1],
        "volume_id": ["valid_1_a_1", "valid_1_a_2", "train_5_a_1"],
        "repo_path": [ctrate_rel_path("valid_1_a_1", "valid"), ctrate_rel_path("valid_1_a_2", "valid"),
                      ctrate_rel_path("train_5_a_1", "train")],
    })


def _metadata():
    return {
        "train": pd.DataFrame({"VolumeName": ["train_5_a_1"], "Rows": [512], "RescaleSlope": [1.0]}),
        "valid": pd.DataFrame({"VolumeName": ["valid_1_a_1", "valid_1_a_2"], "Rows": [512, 512], "RescaleSlope": [1.0, 1.0]}),
    }


def _fetcher(tmp_path, content=None, **kw):
    content = content if content is not None else {p: b"NIFTI" for p in _worklist().repo_path}
    log: list = []
    return CtrateFetcher(_worklist(), _metadata(), SourceConfig(), fake_hf(content, tmp_path, log), **kw), log


def test_chunk_ids_are_zero_padded_in_worklist_order(tmp_path):
    assert _fetcher(tmp_path)[0].chunk_ids() == ["0000", "0001"]


def test_fetch_puts_flat_files_in_scratch_and_leaves_no_hugging_face_cache_behind(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    fetcher, log = _fetcher(tmp_path)
    report = fetcher.fetch("0000", raw, cache_fresh=lambda v: False)
    assert report.failed == {}
    assert sorted(p.name for p in raw.iterdir()) == ["valid_1_a_1.nii.gz", "valid_1_a_2.nii.gz"]  # flat, no repo tree
    assert (raw / "valid_1_a_1.nii.gz").read_bytes() == b"NIFTI"
    assert not (raw / ".hf_cache").exists()  # otherwise every download would be stored twice


def test_already_cached_volumes_are_not_downloaded_again(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    fetcher, log = _fetcher(tmp_path)
    fetcher.fetch("0000", raw, cache_fresh=lambda v: v == "valid_1_a_1")
    assert [Path(f).name for f in log] == ["valid_1_a_2.nii.gz"]
    assert not (raw / "valid_1_a_1.nii.gz").exists()


def test_one_failed_volume_is_reported_but_the_rest_of_the_chunk_still_arrives(tmp_path, monkeypatch):
    monkeypatch.setattr(ctrate, "_sleep", lambda s: None)
    raw = tmp_path / "raw"
    raw.mkdir()
    content = {p: b"N" for p in _worklist().repo_path}
    del content[ctrate_rel_path("valid_1_a_2", "valid")]  # a 404 for this one
    fetcher, _ = _fetcher(tmp_path, content)
    report = fetcher.fetch("0000", raw, cache_fresh=lambda v: False)
    assert list(report.failed) == ["valid_1_a_2"] and "KeyError" in report.failed["valid_1_a_2"]
    assert (raw / "valid_1_a_1.nii.gz").exists()


def test_a_chunk_where_every_download_fails_is_a_fetch_error_not_a_silent_success(tmp_path, monkeypatch):
    monkeypatch.setattr(ctrate, "_sleep", lambda s: None)
    raw = tmp_path / "raw"
    raw.mkdir()
    fetcher, _ = _fetcher(tmp_path, content={})  # e.g. not logged in
    with pytest.raises(FetchError, match="every download in chunk 0000 failed"):
        fetcher.fetch("0000", raw, cache_fresh=lambda v: False)


def test_a_transient_error_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(ctrate, "_sleep", lambda s: None)
    calls = {"n": 0}
    good = fake_hf({p: b"N" for p in _worklist().repo_path}, tmp_path)

    def flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("reset by peer")
        return good(**kw)

    raw = tmp_path / "raw"
    raw.mkdir()
    fetcher = CtrateFetcher(_worklist(), _metadata(), SourceConfig(), flaky, retries=2)
    report = fetcher.fetch("0001", raw, cache_fresh=lambda v: False)
    assert report.failed == {} and (raw / "train_5_a_1.nii.gz").exists()


def test_download_volume_moves_the_file_instead_of_copying_it(tmp_path):
    dest = tmp_path / "raw" / "v.nii.gz"
    dest.parent.mkdir()
    download_volume(fake_hf({"repo/v": b"DATA"}, tmp_path), "o/r", "repo/v", tmp_path / "cache", dest)
    assert dest.read_bytes() == b"DATA"
    assert not list((tmp_path / "cache").rglob("*.nii.gz")) and not [p for p in (tmp_path / "cache").rglob("v") if p.is_file()]


def test_build_rows_covers_the_whole_chunk_with_metadata_and_no_split(tmp_path):
    rows = _fetcher(tmp_path)[0].build_rows("0000", tmp_path)
    assert rows.volume_id.tolist() == ["valid_1_a_1", "valid_1_a_2"]
    assert set(rows.source_split) == {"valid"} and "split" not in rows.columns
    assert rows.RescaleSlope.tolist() == [1.0, 1.0]  # joined from the metadata
    assert rows.scan_path.tolist() == ["valid_1_a_1.nii.gz", "valid_1_a_2.nii.gz"]


def test_an_unknown_chunk_is_a_fetch_error(tmp_path):
    with pytest.raises(FetchError, match="not in the worklist"):
        _fetcher(tmp_path)[0].build_rows("0099", tmp_path)
