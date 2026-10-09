"""Dropdown options read from disk, so new runs, experiments and encoders appear without code changes.

Each provider takes ``ctx`` (the current value of every field) and returns a list of strings. The UI still
lets you type a value that is not in the list.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ct_model.config import DEFAULT_ENCODERS_DIR, load_stage2_config
from ct_model.training.config import DEFAULT_EXPERIMENTS_DIR, load_experiment
from ct_preprocessing.config import DEFAULT_CONFIG_PATH, load_config
from ct_preprocessing.runs import list_runs

KERNEL_CLASSES = ["sharp", "soft", "other"]


def _stems(folder: str) -> list[str]:
    return sorted(p.stem for p in Path(folder).glob("*.yaml"))


def experiments(ctx: dict) -> list[str]:
    return _stems(DEFAULT_EXPERIMENTS_DIR)


def encoders(ctx: dict) -> list[str]:
    return _stems(DEFAULT_ENCODERS_DIR)


def _data_config_path(ctx: dict) -> str:
    """The preprocessing config behind this page: its own --config, or the one a model config points to."""
    if ctx.get("experiment"):
        return load_experiment(ctx["experiment"]).data_config
    config = ctx.get("config")
    if config and "model" in Path(config).parts:
        return load_stage2_config(config).data_config
    return config or DEFAULT_CONFIG_PATH


def sources(ctx: dict) -> list[str]:
    return sorted(load_config(_data_config_path(ctx)).sources)


def runs(ctx: dict) -> list[str]:
    cfg = load_config(_data_config_path(ctx))
    if ctx.get("source"):
        return list_runs(cfg.paths, ctx["source"])
    return sorted({r for s in cfg.sources for r in list_runs(cfg.paths, s)})


def experiment_dirs(ctx: dict) -> list[str]:
    """Finished training runs, newest first (outputs/experiments/index.csv)."""
    index = Path(ctx.get("output_dir") or "outputs/experiments") / "index.csv"
    if not index.is_file():
        return []
    frame = pd.read_csv(index)
    if "finished" in frame:
        frame = frame.sort_values("finished", ascending=False)
    return [d for d in frame["dir"].dropna().astype(str) if Path(d).is_dir()]


def volume_ids(ctx: dict) -> list[str]:
    """Volumes the chosen experiment has predictions for (val, and any evaluated split)."""
    folder = Path(ctx.get("experiment_dir") or "")
    if not ctx.get("experiment_dir") or not folder.is_dir():
        return []
    ids: set[str] = set()
    for csv in [folder / "predictions_val.csv", *folder.glob("eval/*/predictions.csv")]:
        if csv.is_file():
            ids |= set(pd.read_csv(csv, usecols=["volume_id"])["volume_id"].astype(str))
    return sorted(ids)


def labels(ctx: dict) -> list[str]:
    path = Path(ctx.get("experiment_dir") or "") / "labels.json"
    if not ctx.get("experiment_dir") or not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("labels", data) if isinstance(data, dict) else data)


def kernel_classes(ctx: dict) -> list[str]:
    return KERNEL_CLASSES
