"""Archive sources (the local NHRD hospital data): a folder of uploaded
.zip / .tar archives, each holding WHOLE patient folders.

The archives sit in a Google Drive folder (read through rclone) or in any
plain folder path (a mounted drive, a local directory). One archive is one
chunk: it is downloaded, unpacked, preprocessed and deleted before the next
is touched, so the server never holds more than about two archives' worth of
raw data.

Unpacking verifies integrity as a side effect -- zip CRCs are checked while
reading, and a truncated tar or zip fails outright -- so a half-uploaded
archive is rejected (and retried on the next run) instead of being silently
used.
"""
from __future__ import annotations

import shutil
import subprocess
import tarfile
import zipfile
import zlib
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import SourceConfig
from ..dicom_loader import discover_scans
from ..manifest import build_manifest_folder
from .base import FetchError, FetchReport

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".zip", ".tar")


def archive_chunk_id(name: str) -> str | None:
    """'nhrd_A_001.zip' -> 'nhrd_A_001'; None if ``name`` is not a supported archive."""
    lower = name.lower()
    for suffix in ARCHIVE_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return None


# ------------------------------------------------------------------------ backends
class LocalBackend:
    """A plain folder of archives (a mounted drive or any directory)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def list(self) -> list[str]:
        return sorted(p.name for p in self.folder.iterdir() if p.is_file())

    def get(self, name: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.folder / name, dest)


class RcloneBackend:
    """An rclone remote such as ``gdrive:nhrd_raw`` (set up once with ``rclone config``)."""

    def __init__(self, remote: str, runner: Callable = subprocess.run, rclone: str = "rclone"):
        self.remote = remote.rstrip("/")
        self._run = runner
        self._rclone = rclone

    def _call(self, *args: str) -> str:
        try:
            done = self._run([self._rclone, *args], check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise FetchError("rclone is not installed or not on PATH (install it, then run `rclone config`)") from exc
        except subprocess.CalledProcessError as exc:
            raise FetchError(f"rclone {args[0]} failed: {(exc.stderr or exc.stdout or '').strip()[:500]}") from exc
        return done.stdout

    def list(self) -> list[str]:
        return sorted(line.strip() for line in self._call("lsf", self.remote, "--files-only").splitlines() if line.strip())

    def get(self, name: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._call("copyto", f"{self.remote}/{name}", str(dest))


def make_backend(remote: str):
    """A plain existing folder is read directly; anything else is an rclone remote."""
    return LocalBackend(remote) if Path(remote).is_dir() else RcloneBackend(remote)


# ---------------------------------------------------------------------- extraction
def _inside(target: Path, root: Path) -> bool:
    return target == root or root in target.parents


def safe_extract(archive: Path, dest: Path) -> int:
    """Unpack a .zip/.tar(.gz) into ``dest``; returns the number of files written.

    Refuses members that would land outside ``dest`` (absolute paths, ``..``),
    and skips anything that is not a plain file or folder (links, devices).
    """
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        if archive.name.lower().endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    target = (dest / info.filename).resolve()
                    if not _inside(target, dest):
                        raise FetchError(f"unsafe path in archive: {info.filename!r}")
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as out:  # reading to EOF checks the CRC
                        shutil.copyfileobj(src, out)
                    count += 1
        else:
            with tarfile.open(archive, "r:*") as tf:
                for member in tf:
                    target = (dest / member.name).resolve()
                    if not _inside(target, dest):
                        raise FetchError(f"unsafe path in archive: {member.name!r}")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile():
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = tf.extractfile(member)
                    with open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                    count += 1
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, zlib.error, OSError) as exc:
        raise FetchError(f"corrupt or incomplete archive {archive.name}: {exc}") from exc
    if count == 0:
        raise FetchError(f"archive {archive.name} contained no files")
    return count


# ------------------------------------------------------------------------- fetcher
class ArchiveFetcher:
    def __init__(self, backend, source_cfg: SourceConfig):
        self.backend = backend
        self.cfg = source_cfg
        self._names: dict[str, str] = {}  # chunk id -> archive file name

    def chunk_ids(self) -> list[str]:
        self._names = {}
        for name in self.backend.list():
            chunk_id = archive_chunk_id(name)
            if chunk_id is not None:
                self._names[chunk_id] = name
        return sorted(self._names)

    def fetch(self, chunk_id: str, raw_dir: Path, *, cache_fresh: Callable[[str], bool]) -> FetchReport:
        if chunk_id not in self._names:
            self.chunk_ids()
        if chunk_id not in self._names:
            raise FetchError(f"archive for chunk {chunk_id!r} is no longer available")
        name = self._names[chunk_id]
        incoming = raw_dir / ".incoming" / name
        self.backend.get(name, incoming)
        try:
            safe_extract(incoming, raw_dir)
        finally:
            shutil.rmtree(incoming.parent, ignore_errors=True)  # free the archive before preprocessing starts
        return FetchReport()

    def build_rows(self, chunk_id: str, raw_dir: Path, *, patient_id_source: str) -> pd.DataFrame:
        entries = discover_scans(raw_dir, fmt=self.cfg.format)
        if not entries:
            raise FetchError(f"no scans found in archive {chunk_id!r} (format={self.cfg.format})")
        return build_manifest_folder(
            raw_dir, entries, patient_depth=self.cfg.patient_path_depth, patient_id_source=patient_id_source
        )
