"""Where ingest keeps its bookkeeping, and the marker files that make it resumable.

Everything lives under the configured ``paths`` (never hard-coded), one
folder per source, so CT-RATE and NHRD never touch each other's state:

    <worklist_dir>/<source>.csv               what to fetch (CT-RATE only)
    <chunk_manifest_dir>/<source>/chunk_<id>.csv   manifest rows of one chunk
    <state_dir>/<source>/chunk_<id>.done      finished chunk (JSON summary)
    <state_dir>/<source>/chunk_<id>.failed    chunk that could not be processed
    <state_dir>/<source>/patient_id_source.txt   sticky "auto" decision (NHRD)
    <splits_dir>/<source>.csv                 frozen patient -> split

Markers are written atomically (temp file + rename), so a crash can never
leave a half-written marker that makes a chunk look finished when it is not.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import PathsConfig


def chunk_label(chunk) -> str:
    """Integer chunk numbers become zero-padded labels ('0007') so file names
    sort; archive chunk ids (already strings) pass through unchanged."""
    return f"{int(chunk):04d}" if isinstance(chunk, int) else str(chunk)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def worklist_path(paths: PathsConfig, source: str) -> Path:
    return Path(paths.worklist_dir) / f"{source}.csv"


def splits_path(paths: PathsConfig, source: str) -> Path:
    return Path(paths.splits_dir) / f"{source}.csv"


def chunk_manifest_dir(paths: PathsConfig, source: str) -> Path:
    return Path(paths.chunk_manifest_dir) / source


def chunk_manifest_path(paths: PathsConfig, source: str, chunk_id: str) -> Path:
    return chunk_manifest_dir(paths, source) / f"chunk_{chunk_id}.csv"


def state_dir(paths: PathsConfig, source: str) -> Path:
    return Path(paths.state_dir) / source


def _marker(paths: PathsConfig, source: str, chunk_id: str, kind: str) -> Path:
    return state_dir(paths, source) / f"chunk_{chunk_id}.{kind}"


def write_done(paths: PathsConfig, source: str, chunk_id: str, info: dict) -> None:
    write_atomic(_marker(paths, source, chunk_id, "done"), json.dumps(info, indent=1, default=str))
    _marker(paths, source, chunk_id, "failed").unlink(missing_ok=True)


def write_failed(paths: PathsConfig, source: str, chunk_id: str, error: str) -> None:
    attempts = (read_marker(paths, source, chunk_id, "failed") or {}).get("attempts", 0) + 1
    write_atomic(
        _marker(paths, source, chunk_id, "failed"),
        json.dumps({"error": error, "attempts": attempts}, indent=1),
    )


def read_marker(paths: PathsConfig, source: str, chunk_id: str, kind: str) -> dict | None:
    try:
        return json.loads(_marker(paths, source, chunk_id, kind).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def clear_done(paths: PathsConfig, source: str, chunk_id: str) -> None:
    _marker(paths, source, chunk_id, "done").unlink(missing_ok=True)


def _chunks_with(paths: PathsConfig, source: str, kind: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    folder = state_dir(paths, source)
    if folder.is_dir():
        for p in sorted(folder.glob(f"chunk_*.{kind}")):
            chunk_id = p.name[len("chunk_") : -len(f".{kind}")]
            out[chunk_id] = read_marker(paths, source, chunk_id, kind) or {}
    return out


def done_chunks(paths: PathsConfig, source: str) -> dict[str, dict]:
    return _chunks_with(paths, source, "done")


def failed_chunks(paths: PathsConfig, source: str) -> dict[str, dict]:
    return _chunks_with(paths, source, "failed")


def read_patient_id_source(paths: PathsConfig, source: str) -> str | None:
    try:
        value = (state_dir(paths, source) / "patient_id_source.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def write_patient_id_source(paths: PathsConfig, source: str, value: str) -> None:
    write_atomic(state_dir(paths, source) / "patient_id_source.txt", value + "\n")


# --------------------------------------------------------------------- scratch safety
SCRATCH_SENTINEL = ".ingest_scratch"


class ScratchError(RuntimeError):
    """The configured raw_dir holds files ingest did not create."""


def prepare_scratch(raw_dir: Path) -> None:
    """Make ``raw_dir`` ready as ingest's scratch folder and mark it as ours.

    Ingest DELETES everything inside this folder after every chunk, so it must
    never be pointed at real data. A folder is accepted only if it is new,
    empty, or already carries our sentinel file; anything else is refused.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True)
    elif not (raw_dir / SCRATCH_SENTINEL).exists() and any(raw_dir.iterdir()):
        raise ScratchError(
            f"{raw_dir} is not empty and was not created by ingest. Ingest deletes everything inside its "
            "raw_dir after each chunk, so it refuses to use a folder that may hold real data. "
            "Point sources.<name>.raw_dir at a new or empty folder."
        )
    (raw_dir / SCRATCH_SENTINEL).touch()


def clear_scratch(raw_dir: Path) -> None:
    """Empty the scratch folder (keeping the folder and its sentinel)."""
    import shutil

    raw_dir = Path(raw_dir)
    if not (raw_dir / SCRATCH_SENTINEL).exists():
        raise ScratchError(f"refusing to clear {raw_dir}: it was not prepared by prepare_scratch()")
    for child in raw_dir.iterdir():
        if child.name == SCRATCH_SENTINEL:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
