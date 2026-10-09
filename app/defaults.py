"""The value a script really uses when an argument is left out: most arguments default to None on the
command line and fall back to a YAML config, so the UI reads the same configs to show it.

One resolver per script family. A resolver gets ``ctx`` -- the current value of every field (what the
user picked, else its default), so e.g. picking another experiment shows that experiment's lr -- and
returns ``{dest: Default}``. Fields it does not know keep argparse's own default.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from ct_model.config import DEFAULT_ENCODE_CONFIG, load_stage2_config
from ct_model.training.config import experiment_config_path, load_experiment
from ct_preprocessing.config import DEFAULT_CONFIG_PATH, load_config


@dataclass(frozen=True)
class Default:
    value: Any
    origin: str   # where it comes from, shown in the field's tooltip


Resolver = Callable[[dict], dict[str, Default]]


def _data_config(ctx: dict):
    path = ctx.get("config") or DEFAULT_CONFIG_PATH
    return path, load_config(path)


def worklist_defaults(ctx: dict) -> dict[str, Default]:
    path, cfg = _data_config(ctx)
    source = ctx.get("source") or "ctrate"
    ing = cfg.source(source).ingest
    at = f"{path} → sources.{source}.ingest"
    return {
        "source": Default("ctrate", "make_worklist.py"),
        "train_kernel": Default(ing.train_kernel, f"{at}.train_kernel"),
        "chunk_size": Default(ing.chunk_size, f"{at}.chunk_size"),
        "max_train_patients": Default(ing.max_train_patients, f"{at}.max_train_patients"),
        "max_test_patients": Default(ing.max_test_patients, f"{at}.max_test_patients"),
        "max_combined_gb": Default(ing.max_combined_gb, f"{at}.max_combined_gb"),
        "seed": Default(ing.seed, f"{at}.seed"),
    }


def ingest_defaults(ctx: dict) -> dict[str, Default]:
    path, cfg = _data_config(ctx)
    source = ctx.get("source")
    out = {"device": Default(cfg.preprocess.device, f"{path} → preprocess.device")}
    if source in cfg.sources:
        ing, at = cfg.source(source).ingest, f"{path} → sources.{source}.ingest"
        out |= {"min_free_gb": Default(ing.min_free_gb, f"{at}.min_free_gb"),
                "workers": Default(ing.workers, f"{at}.workers"),
                "drive_remote": Default(ing.drive_remote, f"{at}.drive_remote")}
    return out


def splits_defaults(ctx: dict) -> dict[str, Default]:
    path, cfg = _data_config(ctx)
    source = ctx.get("source")
    if source not in cfg.sources:
        return {}
    sp, at = cfg.source(source).split, f"{path} → sources.{source}.split"
    return {"n_val_patients": Default(sp.n_val_patients, f"{at}.n_val_patients"),
            "n_test_patients": Default(sp.n_test_patients, f"{at}.n_test_patients"),
            "seed": Default(sp.seed, f"{at}.seed")}


def kernel_table_defaults(ctx: dict) -> dict[str, Default]:
    path, cfg = _data_config(ctx)
    return {"metadata": Default(f"{cfg.paths.metadata_dir}/train_metadata.csv", f"{path} → paths.metadata_dir"),
            "out": Default(cfg.paths.kernel_table, f"{path} → paths.kernel_table")}


def stage2_defaults(ctx: dict) -> dict[str, Default]:
    path = ctx.get("config") or DEFAULT_ENCODE_CONFIG
    cfg = load_stage2_config(path)
    sel, enc = cfg.selection, cfg.encode
    return {
        "encoder": Default(cfg.encoder, f"{path} → encoder"),
        "source": Default(sel.source, f"{path} → selection.source"),
        "run": Default(sel.run, f"{path} → selection.run (empty = the only run of the source)"),
        "splits": Default(list(sel.splits), f"{path} → selection.splits"),
        "kernel_classes": Default(list(sel.kernel_classes) if sel.kernel_classes else None,
                                  f"{path} → selection.kernel_classes (empty = every kernel class)"),
        "device": Default(enc.device, f"{path} → encode.device"),
        "amp_dtype": Default(enc.amp_dtype, f"{path} → encode.amp_dtype"),
        "slice_batch_size": Default(enc.slice_batch_size, f"{path} → encode.slice_batch_size"),
        "num_workers": Default(enc.num_workers, f"{path} → encode.num_workers"),
        "min_free_gb": Default(enc.min_free_gb, f"{path} → encode.min_free_gb"),
        "limit": Default(enc.limit, f"{path} → encode.limit (empty = all)"),
    }


def train_defaults(ctx: dict) -> dict[str, Default]:
    name = ctx.get("experiment")
    if not name:
        return {}
    cfg, at = load_experiment(name), str(experiment_config_path(name))
    o = cfg.optim
    return {
        "run": Default(cfg.data.run, f"{at} → data.run"),
        "seed": Default(cfg.seed, f"{at} → seed"),
        "loss": Default(cfg.loss.get("type"), f"{at} → loss.type"),
        "lr": Default(o.lr, f"{at} → optim.lr"),
        "weight_decay": Default(o.weight_decay, f"{at} → optim.weight_decay"),
        "batch_size": Default(o.batch_size, f"{at} → optim.batch_size"),
        "max_epochs": Default(o.max_epochs, f"{at} → optim.max_epochs"),
        "patience": Default(o.patience, f"{at} → optim.patience"),
        "device": Default(cfg.device, f"{at} → device"),
        "output_dir": Default(cfg.output_dir, f"{at} → output_dir"),
    }


def trained_run_defaults(ctx: dict) -> dict[str, Default]:
    """evaluate / explain: the source and run the experiment was trained on (its saved config.yaml)."""
    folder = ctx.get("experiment_dir")
    cfg_file = Path(folder) / "config.yaml" if folder else None
    if not cfg_file or not cfg_file.is_file():
        return {}
    data = (yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}).get("data", {})
    return {"source": Default(data.get("source"), f"{cfg_file} → data.source"),
            "run": Default(data.get("run"), f"{cfg_file} → data.run")}
