"""Runs: one named folder per worklist and everything derived from it.

A run is a plan ("download the sharp reconstruction of these patients") plus its
outputs. Runs coexist -- nothing is ever overwritten or deleted -- while the cache of
preprocessed volumes is shared and only grows:

    data/cache/                       {volume_id}.npy + .meta.json     (shared by every run)
    data/splits/<source>.csv          frozen patient -> split          (shared: a patient never changes split)
    data/runs/<source>/<name>/
        run.json                      the settings that made this run
        worklist.csv                  what to download (CT-RATE; archive sources list Drive instead)
        chunk_manifests/ state/       per-chunk rows and done/failed markers
        manifest.csv  qc_report.csv  qc_montages/  preprocessing_manifest.json  splits.csv

Every script takes ``--run NAME``. With one run on disk it is picked automatically.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import PathsConfig

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_ARCHIVE_RUN = "main"


@dataclass(frozen=True)
class Run:
    paths: PathsConfig
    source: str
    name: str

    @property
    def dir(self) -> Path:
        return Path(self.paths.runs_dir) / self.source / self.name

    @property
    def info_path(self) -> Path:
        return self.dir / "run.json"

    @property
    def worklist_path(self) -> Path:
        return self.dir / "worklist.csv"

    @property
    def chunk_manifest_dir(self) -> Path:
        return self.dir / "chunk_manifests"

    def chunk_manifest_path(self, chunk_id: str) -> Path:
        return self.chunk_manifest_dir / f"chunk_{chunk_id}.csv"

    @property
    def state_dir(self) -> Path:
        return self.dir / "state"

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.csv"

    @property
    def qc_report_path(self) -> Path:
        return self.dir / "qc_report.csv"

    @property
    def montage_dir(self) -> Path:
        return self.dir / "qc_montages"

    @property
    def record_path(self) -> Path:
        return self.dir / "preprocessing_manifest.json"

    @property
    def splits_copy_path(self) -> Path:
        return self.dir / "splits.csv"


def check_run_name(name: str) -> str:
    if not _NAME.match(name or ""):
        raise ValueError(f"invalid run name {name!r}: use letters, digits, '.', '_' or '-' (e.g. train-sharp)")
    return name


def list_runs(paths: PathsConfig, source: str) -> list[str]:
    root = Path(paths.runs_dir) / source
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def new_run(paths: PathsConfig, source: str, name: str) -> Run:
    """A run that does not exist yet. An existing one is never touched."""
    run = Run(paths, source, check_run_name(name))
    if run.dir.exists():
        raise FileExistsError(
            f"run {name!r} of source {source!r} already exists ({run.dir}) -- runs are never overwritten. "
            f"Use it as is (--run {name}) or pick another name with --name."
        )
    run.dir.mkdir(parents=True)
    return run


def get_run(paths: PathsConfig, source: str, name: str) -> Run:
    run = Run(paths, source, check_run_name(name))
    if not run.dir.is_dir():
        have = list_runs(paths, source)
        raise FileNotFoundError(
            f"no run {name!r} for source {source!r}" + (f" (existing runs: {', '.join(have)})" if have else
            " -- there are no runs yet; run scripts/preprocessing/make_worklist.py first")
        )
    return run


def resolve_run(paths: PathsConfig, source: str, name: str | None = None, *, create_default: str | None = None) -> Run:
    """The run a script should work on: ``name`` if given, else the only run that exists.

    ``create_default`` (archive sources, which have no worklist) creates a run of that name
    when none exists yet.
    """
    if name:
        run = Run(paths, source, check_run_name(name))
        if create_default and not run.dir.exists() and name == create_default:
            run.dir.mkdir(parents=True)
        return get_run(paths, source, name)
    have = list_runs(paths, source)
    if len(have) == 1:
        return get_run(paths, source, have[0])
    if not have:
        if create_default:
            return resolve_run(paths, source, create_default, create_default=create_default)
        raise FileNotFoundError(
            f"no runs for source {source!r} yet -- run scripts/preprocessing/make_worklist.py first"
        )
    raise ValueError(f"source {source!r} has several runs ({', '.join(have)}) -- choose one with --run NAME")


def write_run_info(run: Run, info: dict) -> None:
    from .ingest.state import write_atomic

    info = {"name": run.name, "source": run.source, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}
    write_atomic(run.info_path, json.dumps(info, indent=2, default=str))


def read_run_info(run: Run) -> dict:
    try:
        return json.loads(run.info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def auto_run_name(train_kernel: str) -> str:
    """The default name of a run: what it trains on."""
    return f"train-{train_kernel}"
