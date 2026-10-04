"""Archive sources: backends, safe unpacking, and the fetcher (NHRD)."""
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from ct_preprocessing.config import SourceConfig
from ct_preprocessing.ingest.archives import (
    ArchiveFetcher, LocalBackend, RcloneBackend, archive_chunk_id, make_backend, safe_extract,
)
from ct_preprocessing.ingest.base import FetchError


def make_zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return path


def zip_folder(folder: Path, archive: Path, root: Path | None = None) -> Path:
    root = root or folder
    with zipfile.ZipFile(archive, "w") as zf:
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(root).as_posix())
    return archive


NHRD = SourceConfig(format="dicom", manifest_builder="folder", patient_id_source="path", patient_path_depth=1)


def test_archive_chunk_id_strips_the_supported_suffixes():
    assert archive_chunk_id("nhrd_A_001.zip") == "nhrd_A_001"
    assert archive_chunk_id("x.tar") == "x" and archive_chunk_id("x.tar.gz") == "x" and archive_chunk_id("x.TGZ") == "x"
    assert archive_chunk_id("patients_on_disk.txt") is None
    assert archive_chunk_id("notes.zip.bak") is None


# ---------------------------------------------------------------------- backends
def test_local_backend_lists_files_and_copies_them(tmp_path):
    (tmp_path / "drive").mkdir()
    (tmp_path / "drive" / "a.zip").write_bytes(b"1")
    (tmp_path / "drive" / "sub").mkdir()
    b = LocalBackend(tmp_path / "drive")
    assert b.list() == ["a.zip"]  # folders are not archives
    b.get("a.zip", tmp_path / "out" / "a.zip")
    assert (tmp_path / "out" / "a.zip").read_bytes() == b"1"


def test_make_backend_reads_a_real_folder_directly_and_treats_anything_else_as_rclone(tmp_path):
    assert isinstance(make_backend(str(tmp_path)), LocalBackend)
    assert isinstance(make_backend("gdrive:nhrd_raw"), RcloneBackend)


class FakeRclone:
    def __init__(self, stdout="", error=None):
        self.calls, self.stdout, self.error = [], stdout, error

    def __call__(self, cmd, **kw):
        self.calls.append(cmd)
        if self.error:
            raise self.error
        return subprocess.CompletedProcess(cmd, 0, stdout=self.stdout, stderr="")


def test_rclone_backend_lists_and_copies_with_the_right_commands(tmp_path):
    run = FakeRclone(stdout="b.zip\na.zip\n\n")
    b = RcloneBackend("gdrive:nhrd_raw/", runner=run)
    assert b.list() == ["a.zip", "b.zip"]
    assert run.calls[0] == ["rclone", "lsf", "gdrive:nhrd_raw", "--files-only"]
    b.get("a.zip", tmp_path / "x" / "a.zip")
    assert run.calls[1] == ["rclone", "copyto", "gdrive:nhrd_raw/a.zip", str(tmp_path / "x" / "a.zip")]


def test_rclone_failures_become_fetch_errors_with_a_useful_message():
    err = subprocess.CalledProcessError(1, "rclone", stderr="Failed to create file system: didn't find section")
    with pytest.raises(FetchError, match="didn't find section"):
        RcloneBackend("gdrive:x", runner=FakeRclone(error=err)).list()
    with pytest.raises(FetchError, match="rclone is not installed"):
        RcloneBackend("gdrive:x", runner=FakeRclone(error=FileNotFoundError())).list()


# --------------------------------------------------------------------- extraction
def test_safe_extract_unpacks_zip_tar_and_targz_keeping_the_folder_structure(tmp_path):
    src = tmp_path / "src"
    (src / "4203-26" / "P1").mkdir(parents=True)
    (src / "4203-26" / "P1" / "a.dcm").write_bytes(b"AAAA")
    (src / "4214-26").mkdir()
    (src / "4214-26" / "b.dcm").write_bytes(b"BB")

    zip_folder(src, tmp_path / "x.zip")
    for mode, name in (("w", "x.tar"), ("w:gz", "x.tar.gz")):
        with tarfile.open(tmp_path / name, mode) as tf:
            tf.add(src / "4203-26", arcname="4203-26")
            tf.add(src / "4214-26", arcname="4214-26")

    for i, name in enumerate(("x.zip", "x.tar", "x.tar.gz")):
        out = tmp_path / f"out{i}"
        assert safe_extract(tmp_path / name, out) == 2
        assert (out / "4203-26" / "P1" / "a.dcm").read_bytes() == b"AAAA"
        assert (out / "4214-26" / "b.dcm").read_bytes() == b"BB"


def test_a_path_traversal_member_is_refused_and_writes_nothing_outside(tmp_path):
    bad = make_zip(tmp_path / "evil.zip", {"../escaped.txt": b"x"})
    with pytest.raises(FetchError, match="unsafe path"):
        safe_extract(bad, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_a_truncated_zip_is_rejected_instead_of_half_used(tmp_path):
    good = make_zip(tmp_path / "g.zip", {"a/x.dcm": b"0" * 5000, "a/y.dcm": b"1" * 5000})
    cut = tmp_path / "cut.zip"
    cut.write_bytes(good.read_bytes()[: good.stat().st_size // 2])  # a half-finished upload
    with pytest.raises(FetchError, match="corrupt or incomplete"):
        safe_extract(cut, tmp_path / "out")


def test_a_truncated_tar_is_rejected(tmp_path):
    full = tmp_path / "f.tar"
    with tarfile.open(full, "w") as tf:
        (tmp_path / "big.bin").write_bytes(b"0" * 200_000)
        tf.add(tmp_path / "big.bin", arcname="p/big.bin")
    cut = tmp_path / "cut.tar"
    cut.write_bytes(full.read_bytes()[:100_000])
    with pytest.raises(FetchError, match="corrupt or incomplete"):
        safe_extract(cut, tmp_path / "out")


def test_an_archive_with_no_files_is_an_error(tmp_path):
    with pytest.raises(FetchError, match="no files"):
        safe_extract(make_zip(tmp_path / "e.zip", {}), tmp_path / "out")


# ------------------------------------------------------------------------ fetcher
def test_chunk_ids_are_the_archives_on_the_drive_and_ignore_everything_else(tmp_path):
    drive = tmp_path / "drive"
    drive.mkdir()
    for name in ("nhrd_B_001.zip", "nhrd_A_002.tar", "nhrd_A_001.zip", "patients_on_disk.txt", "readme.md"):
        (drive / name).write_bytes(b"x")
    assert ArchiveFetcher(LocalBackend(drive), NHRD).chunk_ids() == ["nhrd_A_001", "nhrd_A_002", "nhrd_B_001"]


def test_fetch_unpacks_into_scratch_and_removes_the_archive_copy(tmp_path):
    drive, raw = tmp_path / "drive", tmp_path / "raw"
    drive.mkdir()
    raw.mkdir()
    make_zip(drive / "c1.zip", {"4203-26/P1/S1/a.dcm": b"123"})
    fetcher = ArchiveFetcher(LocalBackend(drive), NHRD)
    fetcher.chunk_ids()

    report = fetcher.fetch("c1", raw, cache_fresh=lambda v: False)

    assert report.failed == {}
    assert (raw / "4203-26" / "P1" / "S1" / "a.dcm").read_bytes() == b"123"
    assert not (raw / ".incoming").exists()  # the 10 GB archive is freed before preprocessing starts


def test_fetching_a_corrupt_archive_raises_fetch_error_and_leaves_no_archive_behind(tmp_path):
    drive, raw = tmp_path / "drive", tmp_path / "raw"
    drive.mkdir()
    raw.mkdir()
    (drive / "bad.zip").write_bytes(b"this is not a zip")
    fetcher = ArchiveFetcher(LocalBackend(drive), NHRD)
    fetcher.chunk_ids()
    with pytest.raises(FetchError):
        fetcher.fetch("bad", raw, cache_fresh=lambda v: False)
    assert not (raw / ".incoming").exists()


def test_build_rows_finds_every_series_and_groups_patients_by_top_folder(tmp_path, dicom_writer, synthetic_hu):
    raw = tmp_path / "raw"
    for top in ("4203-26", "4214-26"):
        dicom_writer(raw / top / "P00001" / "S0001", synthetic_hu[:6])
    rows = ArchiveFetcher(LocalBackend(tmp_path), NHRD).build_rows("c1", raw, patient_id_source="path")
    assert sorted(rows.patient_id) == ["4203-26", "4214-26"]
    assert rows.volume_id.is_unique and set(rows.format) == {"dicom"}
    assert rows.attrs["patient_id_source"] == "path"


def test_build_rows_on_an_archive_without_scans_is_an_error(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "notes.txt").write_text("hello")
    with pytest.raises(FetchError, match="no scans found"):
        ArchiveFetcher(LocalBackend(tmp_path), NHRD).build_rows("c1", raw, patient_id_source="path")
