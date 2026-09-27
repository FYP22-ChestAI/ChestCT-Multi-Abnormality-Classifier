"""Copy a scan tree from a slow location (a mounted Google Drive, a network
share) onto fast local disk before processing.

Thousands of small DICOM files read straight from a Drive mount are very
slow; copying them once to the machine's local disk first is much faster.
The copy is resumable (a file already present with the same size is
skipped), retried on transient errors, and can be limited to some top-level
folders. Nothing here is specific to Drive -- ``src`` is any folder path.
"""
from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _copy_one(src: Path, dst: Path, retries: int = 3) -> str:
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            shutil.copyfile(src, dst)
            return "copied"
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return "copied"


def stage_tree(
    src: str | Path,
    dst: str | Path,
    only_folders: list[str] | None = None,
    workers: int = 8,
) -> dict:
    """Copy ``src`` to ``dst`` (keeping the relative structure). Returns counts."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"source folder does not exist: {src}")

    wanted = [f.strip().strip("/") for f in (only_folders or []) if f.strip()]
    jobs = []
    for path in sorted(src.rglob("*")):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(src).parts):
            continue
        rel = path.relative_to(src)
        rel_posix = rel.as_posix()
        if wanted and not any(rel_posix == w or rel_posix.startswith(w + "/") for w in wanted):
            continue
        jobs.append((path, dst / rel))

    if not jobs:
        raise FileNotFoundError(f"no files to copy under {src}" + (f" for folders {wanted}" if wanted else ""))

    counts = {"copied": 0, "skipped": 0, "files": len(jobs)}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for outcome in ex.map(lambda job: _copy_one(*job), jobs):
            counts[outcome] += 1
    counts["bytes"] = sum(d.stat().st_size for _, d in jobs)
    return counts
