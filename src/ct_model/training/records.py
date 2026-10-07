"""Where experiment results go, and how they stay identifiable for later comparisons.

    outputs/experiments/
        index.csv                               one row per FINISHED training run (append-only ledger)
        evaluations.csv                         one row per evaluation (test, or another run)
        <name>/<config hash8>/seed<k>/          one training run -- never overwritten
            config.yaml  run_info.json  labels.json  store_record.json
            checkpoints/best.pt  checkpoints/last.pt
            metrics.csv  predictions_val.csv  thresholds.json  val_metrics.json  val_per_label.csv
            eval/<source>__<run>__<split>/      written by evaluate_mil.py
            explain/<volume_id>/                written by explain_volume.py

The CONFIG HASH identifies what was learned: the resolved experiment config without the settings that
do not change it (name, seed, device, paths -- see training.config.NOT_HASHED), plus the embedding
store (encoder + weights + preprocessing, via its fingerprinted folder name) and the sha256 of the run
manifest (which volumes, which splits, which labels). Same hash = same experiment, so seeds of one
config sit side by side and can be averaged; any change gives a new hash and a new folder.
The run id is ``<name>/<hash8>/seed<k>``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ct_preprocessing.ingest.lock import AlreadyRunning, SourceLock
from ct_preprocessing.ingest.state import write_atomic

from .config import NOT_HASHED, ExperimentConfig

TRAIN_LEDGER = "index.csv"
EVAL_LEDGER = "evaluations.csv"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_hash(cfg: ExperimentConfig, store_name: str, manifest_sha256: str) -> str:
    settings = cfg.to_dict()
    for key in NOT_HASHED:
        settings.pop(key, None)
    payload = {"config": settings, "embedding_store": store_name, "manifest_sha256": manifest_sha256}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def run_id(cfg: ExperimentConfig, cfg_hash: str) -> str:
    return f"{cfg.name}/{cfg_hash[:8]}/seed{cfg.seed}"


def run_dir(cfg: ExperimentConfig, cfg_hash: str) -> Path:
    return Path(cfg.output_dir) / cfg.name / cfg_hash[:8] / f"seed{cfg.seed}"


def git_info(repo: str | Path = ".") -> dict:
    """Commit and whether the working tree had uncommitted changes (None if git is unavailable)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def environment() -> dict:
    import numpy
    import sklearn
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__,
            "sklearn": sklearn.__version__, "gpu": gpu, "host": platform.node()}


def write_json(path: str | Path, obj) -> None:
    write_atomic(Path(path), json.dumps(obj, indent=2, default=str))


def read_json(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def append_row(ledger: str | Path, row: dict, timeout_s: float = 60.0) -> None:
    """Append one row to a CSV ledger (new columns are added, never removed). Several runs may finish
    at the same time, so the read-modify-write happens under a file lock."""
    ledger = Path(ledger)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with SourceLock(ledger.with_name(ledger.name + ".lock"), ledger.name):
                frame = pd.read_csv(ledger) if ledger.is_file() else pd.DataFrame()
                frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
                write_atomic(ledger, frame.to_csv(index=False, lineterminator="\n"))
                return
        except AlreadyRunning:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.2)


def ledger_path(output_dir: str | Path, name: str) -> Path:
    return Path(output_dir) / name


def is_finished(folder: Path) -> bool:
    return read_json(folder / "run_info.json").get("status") == "finished"


def ensure_dir(path: Path) -> Path:
    os.makedirs(path, exist_ok=True)
    return path
