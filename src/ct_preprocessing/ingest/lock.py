"""One ingest per source at a time.

Two runs on the same source would share one scratch folder and one set of
markers -- each would delete the other's raw files mid-chunk. The lock is an OS
file lock held for the life of the process, so it is released automatically if
the run crashes or is killed: there is never a stale lock to clean up by hand.
(Different sources have different lock files, so CT-RATE and NHRD can run side
by side.)
"""
from __future__ import annotations

import os
from pathlib import Path


class AlreadyRunning(RuntimeError):
    """Another ingest for this source holds the lock."""


class SourceLock:
    def __init__(self, path: Path, source: str):
        self.path, self.source, self._fd = Path(path), source, None

    def __enter__(self) -> "SourceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._fd)
            self._fd = None
            raise AlreadyRunning(
                f"another ingest for source {self.source!r} is already running (lock: {self.path}). "
                "Wait for it, or attach to its tmux session; running two at once would delete each other's raw data."
            ) from exc
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            os.close(self._fd)  # closing the descriptor releases the lock
            self._fd = None
