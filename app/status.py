"""Is a pipeline step already done? Each check looks only at what earlier runs saved, for the exact command
the step would run, so a step whose outputs exist is skipped instead of redone.

The checks follow the content, not timestamps alone: new volumes in a run make merge, QC, splits and
encode "to do" again, and a new manifest changes the training config hash, so models are retrained.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from app.commands import parse, script_module
from ct_preprocessing.config import load_config
from ct_preprocessing.ingest import state
from ct_preprocessing.runs import Run, auto_run_name, list_runs

DONE, PARTIAL, TODO, BLOCKED, NA, ALWAYS = "done", "partial", "todo", "blocked", "n/a", "always"


@dataclass(frozen=True)
class Status:
    state: str      # DONE | PARTIAL | TODO | BLOCKED | NA | ALWAYS
    detail: str

    @property
    def needs_run(self) -> bool:
        return self.state in (PARTIAL, TODO, BLOCKED, ALWAYS)


# ------------------------------------------------------------------ helpers
def _run(args) -> Run | None:
    cfg = load_config(args.config)
    name = getattr(args, "run", None) or getattr(args, "name", None)
    if not name:
        have = list_runs(cfg.paths, args.source)
        name = have[0] if len(have) == 1 else None
    return Run(cfg.paths, args.source, name) if name else None


@lru_cache(maxsize=32)
def _csv(path: str, mtime: float, usecols: tuple[str, ...]) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda c: c in usecols)


def _read(path: Path, *cols: str) -> pd.DataFrame | None:
    return _csv(str(path), path.stat().st_mtime, cols) if path.is_file() else None


@lru_cache(maxsize=32)
def _sha256(path: str, mtime: float) -> str:
    from ct_model.training.records import file_sha256
    return file_sha256(path)


def _no_run(args) -> Status:
    return Status(BLOCKED, f"run {getattr(args, 'run', None) or '?'} does not exist yet (planned by the first step)")


# ------------------------------------------------------------------ preprocessing
def worklist(argv: list[str]) -> Status:
    args = parse("scripts/preprocessing/make_worklist.py", argv)
    cfg = load_config(args.config)
    if cfg.source(args.source).manifest_builder != "ctrate":
        return Status(NA, "archive sources have no worklist: ingest lists the archives itself")
    name = args.name or ("kernel-survey" if args.survey else auto_run_name(args.train_kernel or
                                                                           cfg.source(args.source).ingest.train_kernel))
    run = Run(cfg.paths, args.source, name)
    if run.worklist_path.is_file():
        n = len(_read(run.worklist_path, "volume_id"))
        return Status(DONE, f"run {name} exists: {n} volumes planned")
    return Status(TODO, f"run {name} will be created")


def ingest(argv: list[str]) -> Status:
    args = parse("scripts/preprocessing/ingest.py", argv)
    run = _run(args)
    if run is None or not run.dir.is_dir():
        return _no_run(args)
    done = state.done_chunks(run)
    wl = _read(run.worklist_path, "chunk")
    if wl is None:  # archive source: the archive list is on the Drive, ingest skips finished ones itself
        return Status(ALWAYS, f"{len(done)} archives done so far; ingest checks the Drive folder for new ones")
    total = wl["chunk"].nunique()
    failed = state.failed_chunks(run)
    if len(done) >= total:
        return Status(DONE, f"all {total} chunks ingested")
    extra = f", {len(failed)} failed" if failed else ""
    return Status(PARTIAL if done else TODO, f"{len(done)} / {total} chunks ingested{extra} (resumes)")


def merge(argv: list[str]) -> Status:
    args = parse("scripts/preprocessing/merge_manifests.py", argv)
    run = _run(args)
    if run is None or not run.dir.is_dir():
        return _no_run(args)
    markers = list(run.state_dir.glob("*.done")) if run.state_dir.is_dir() else []
    if not markers:
        return Status(BLOCKED, "nothing ingested yet")
    if not run.manifest_path.is_file():
        return Status(TODO, "no manifest.csv yet")
    if run.manifest_path.stat().st_mtime < max(m.stat().st_mtime for m in markers):
        return Status(TODO, "chunks finished after the last merge")
    return Status(DONE, f"manifest.csv has {len(_read(run.manifest_path, 'volume_id'))} volumes")


def qc(argv: list[str]) -> Status:
    args = parse("scripts/preprocessing/qc_report.py", argv)
    run = _run(args)
    manifest = _read(run.manifest_path, "volume_id") if run else None
    if manifest is None:
        return Status(BLOCKED, "waiting for the run's manifest")
    report = _read(run.qc_report_path, "volume_id", "passed")
    if report is None:
        return Status(TODO, "no qc_report.csv yet")
    missing = len(set(manifest.volume_id) - set(report.volume_id))
    if missing:
        return Status(TODO, f"{missing} volumes not checked yet")
    failed = int((~report["passed"].astype(str).str.lower().isin(["true", "1"])).sum())
    return Status(DONE, f"{len(report)} volumes checked, {failed} failed")


def splits(argv: list[str]) -> Status:
    args = parse("scripts/preprocessing/assign_splits.py", argv)
    run = _run(args)
    manifest = _read(run.manifest_path, "split") if run else None
    if manifest is None:
        return Status(BLOCKED, "waiting for the run's manifest")
    counts = manifest["split"].value_counts().to_dict()
    if counts.get("unassigned"):
        return Status(TODO, f"{counts['unassigned']} volumes not assigned yet")
    return Status(DONE, ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))


# ------------------------------------------------------------------ model
def encode(argv: list[str]) -> Status:
    module = script_module("scripts/model/encode_volumes.py")
    settings, data_cfg, enc_cfg = module.settings_from_args(module.build_parser().parse_args(argv))
    run = Run(data_cfg.paths, settings["source"], settings["run"]) if settings["run"] else None
    manifest = _read(run.manifest_path, "volume_id", "split", "qc_passed", "kernel_class") if run else None
    if manifest is None:
        return Status(BLOCKED, "waiting for the run's manifest")
    rows = manifest[manifest["split"].isin(settings["splits"])]
    if settings["qc_passed_only"] and "qc_passed" in rows:
        rows = rows[rows["qc_passed"].astype(str).str.lower().isin(["true", "1"])]
    if settings["kernel_classes"] and "kernel_class" in rows:
        rows = rows[rows["kernel_class"].isin(settings["kernel_classes"])]
    wanted = set(rows["volume_id"])
    folder = Path(settings["embeddings_dir"]) / enc_cfg.store_name
    have = {p.name[:-4] for p in folder.glob("*.npy")} if folder.is_dir() else set()
    n_done = len(wanted & have)
    if not wanted:
        return Status(BLOCKED, "no volumes selected yet (splits not assigned?)")
    if n_done == len(wanted):
        return Status(DONE, f"all {n_done} volumes in {folder.name}")
    return Status(PARTIAL if n_done else TODO, f"{n_done} / {len(wanted)} volumes encoded (resumes)")


def train_run_dir(argv: list[str]) -> Path | None:
    """Where train_mil.py writes this command's run, or None while the run's manifest does not exist."""
    from ct_model.config import load_encoder_config
    from ct_model.training.records import config_hash, run_dir

    module = script_module("scripts/model/train_mil.py")
    cfg = module.experiment_from_args(module.build_parser().parse_args(argv))
    data_cfg = load_config(cfg.data_config)
    names = [cfg.data.run] if cfg.data.run else list_runs(data_cfg.paths, cfg.data.source)
    if len(names) != 1:
        return None
    manifest = Run(data_cfg.paths, cfg.data.source, names[0]).manifest_path
    if not manifest.is_file():
        return None
    store = load_encoder_config(cfg.encoder).store_name
    return run_dir(cfg, config_hash(cfg, store, _sha256(str(manifest), manifest.stat().st_mtime)))


def train(argv: list[str]) -> Status:
    folder = train_run_dir(argv)
    if folder is None:
        return Status(BLOCKED, "waiting for the run's manifest")
    info = _json(folder / "run_info.json")
    if info.get("status") == "finished":
        auroc = _json(folder / "val_metrics.json").get("macro_auroc")
        return Status(DONE, f"trained: best epoch {info.get('best_epoch')}"
                            + (f", val macro AUROC {auroc:.4f}" if auroc is not None else ""))
    if (folder / "checkpoints" / "last.pt").is_file():
        return Status(PARTIAL, "interrupted: resumes from its last epoch")
    return Status(TODO, f"will train into {folder.as_posix()}")


def evaluate(argv: list[str]) -> Status:
    args = parse("scripts/model/evaluate_mil.py", argv)
    folder = Path(args.experiment_dir)
    cfg_file = folder / "config.yaml"
    if not cfg_file.is_file():
        return Status(BLOCKED, "after its training run")
    data = (yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}).get("data", {})
    out = folder / "eval" / f"{args.source or data.get('source')}__{args.run or data.get('run')}__{args.split}"
    if out.is_dir() and not args.overwrite:
        summary = _read(out / "metrics_summary.csv", "group_by", "macro_auroc")
        auroc = summary[summary.group_by == "all"].macro_auroc.iloc[0] if summary is not None and len(summary) else None
        return Status(DONE, f"evaluated on {args.split}" + (f": macro AUROC {auroc:.4f}" if auroc is not None else ""))
    return Status(TODO, f"evaluate on {args.split}" + (" (redo)" if out.is_dir() else ""))


def always(argv: list[str]) -> Status:
    return Status(ALWAYS, "quick: always refreshed at the end")


def _json(path: Path) -> dict:
    import json
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
