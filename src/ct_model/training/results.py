"""Read every experiment under outputs/experiments/ back into tables, for reports and notebooks.

    from ct_model.training.results import load_results
    res = load_results("outputs/experiments")
    res.runs          one row per run folder: run id, config, status, best epoch, val metrics
    res.curves        metrics.csv of every run (one row per run and epoch)
    res.val_labels    val_per_label.csv of every run (its ``label`` column is renamed ``finding``)
    res.test          metrics_summary.csv of every evaluation (one row per run, eval and group)
    res.test_labels   metrics_per_label.csv of every evaluation
    res.clinical      scorecard.json headline of every clinical report (one row per run or ensemble and eval)
    res.clinical_labels  per_label.csv of every clinical report, with the gate status of each label

Everything is found by walking the run folders (``<name>/<hash8>/seed<k>/``), not by reading the
ledgers, so it works from any folder that holds runs -- the whole output dir, one experiment name,
or one config hash -- and still shows runs that crashed or are still training (``status``).

Each configuration also gets a short readable ``setup`` label: the settings that differ between the
configs found (e.g. ``loss=asl``, ``max_epochs=30``), so two configs of the same name can be told apart
without opening their config.yaml files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .records import read_json

# settings that never tell two configs apart in a useful way
_NOT_A_SETTING = {"name", "seed", "device", "output_dir", "data_config", "encode_config"}
# short names for the settings that usually differ (others keep their last key)
_SHORT = {"aggregator.type": "aggregator", "aggregator.branches": "branches", "loss.type": "loss",
          "optim.max_epochs": "max_epochs", "optim.lr": "lr", "head.type": "head"}


@dataclass
class Results:
    runs: pd.DataFrame
    curves: pd.DataFrame
    val_labels: pd.DataFrame
    test: pd.DataFrame
    test_labels: pd.DataFrame
    settings: pd.DataFrame  # one row per config hash, one column per setting that differs between configs
    clinical: pd.DataFrame = field(default_factory=pd.DataFrame)
    clinical_labels: pd.DataFrame = field(default_factory=pd.DataFrame)


def flatten(d: dict, prefix: str = "") -> dict:
    """{"optim": {"lr": 1e-3}} -> {"optim.lr": 1e-3}; lists become strings so they can be compared."""
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = str(value) if isinstance(value, (list, tuple)) else value
    return out


def find_run_dirs(root: str | Path) -> list[Path]:
    """Every run folder (one with a config.yaml) under ``root``, including ``root`` itself."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist -- train something with scripts/model/train_mil.py first")
    return sorted(p.parent for p in root.rglob("config.yaml") if (p.parent / "run_info.json").is_file())


def _read_csv(path: Path, **cols) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path).rename(columns={"label": "finding"})  # per-label files: the finding's name
    for key, value in cols.items():
        frame.insert(0, key, value)
    return frame


def _setup_labels(configs: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Settings that differ between config hashes, and a short label per hash.

    The label lists only the settings where a config departs from the most common value
    (``loss=asl``, ``max_epochs=30, patience=30``); a config that departs from none is the
    ``reference``. A setting only some configs have (``attn_dim`` of an attention aggregator)
    counts only when its values differ, since it already follows from the setting that does.
    """
    configs = configs.drop(columns=[c for c in configs.columns if c in _NOT_A_SETTING], errors="ignore").astype(object)
    varying = [c for c in configs.columns if configs[c].dropna().astype(str).nunique() > 1]
    settings = configs[varying]
    if settings.empty:
        return settings, pd.Series("reference", index=configs.index)
    common = {c: settings[c].dropna().astype(str).mode().iloc[0] for c in varying}

    def label(row: pd.Series) -> str:
        parts = [f"{_SHORT.get(col, col.rsplit('.', 1)[-1])}={value}" for col, value in row.items()
                 if pd.notna(value) and str(value) != common[col]]
        return ", ".join(parts) or "reference"

    return settings, settings.apply(label, axis=1)


def load_results(root: str | Path = "outputs/experiments") -> Results:
    rows, configs, curves, val_labels, test, test_labels = [], {}, [], [], [], []
    for folder in find_run_dirs(root):
        info = read_json(folder / "run_info.json")
        cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8")) or {}
        rid = info.get("run_id") or folder.as_posix()
        cfg_hash = info.get("config_hash", folder.parent.name)
        configs.setdefault(cfg_hash, flatten(cfg))
        optim = cfg.get("optim", {})
        rows.append({
            "run_id": rid, "name": cfg.get("name"), "config_hash": cfg_hash, "hash8": cfg_hash[:8],
            "seed": cfg.get("seed"), "status": info.get("status", "unknown"), "started": info.get("started"),
            "ended": info.get("ended"), "stopped": info.get("stopped"), "best_epoch": info.get("best_epoch"),
            "epochs_run": info.get("epochs_run"), "max_epochs": optim.get("max_epochs"),
            "patience": optim.get("patience"), "steps_per_epoch": info.get("steps_per_epoch"),
            "git_commit": (info.get("git") or {}).get("commit"), "git_dirty": (info.get("git") or {}).get("dirty"),
            **{k if k.startswith("val_") else f"val_{k}": v
               for k, v in read_json(folder / "val_metrics.json").items() if k != "n_volumes"},
            "dir": folder.as_posix(),
        })
        ids = {"run_id": rid, "config_hash": cfg_hash}
        for frame, out in ((_read_csv(folder / "metrics.csv", **ids), curves),
                           (_read_csv(folder / "val_per_label.csv", **ids), val_labels)):
            if frame is not None:
                out.append(frame)
        for ev in sorted((folder / "eval").glob("*")) if (folder / "eval").is_dir() else []:
            source, run, split = (ev.name.split("__") + ["", "", ""])[:3]
            eids = {**ids, "eval": ev.name, "eval_split": split}
            for frame, out in ((_read_csv(ev / "metrics_summary.csv", **eids), test),
                               (_read_csv(ev / "metrics_per_label.csv", **eids), test_labels)):
                if frame is not None:
                    out.append(frame)

    runs = pd.DataFrame(rows)
    if runs.empty:
        raise FileNotFoundError(f"no run folders (config.yaml + run_info.json) under {root}")
    settings, setup = _setup_labels(pd.DataFrame.from_dict(configs, orient="index"))
    runs["setup"] = runs.config_hash.map(setup)
    runs["config"] = runs.name + " [" + runs.hash8 + "]"

    def cat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=["run_id", "config_hash"])
        frame = pd.concat(frames, ignore_index=True)
        return frame.merge(runs[["run_id", "name", "hash8", "config", "setup", "seed"]], on="run_id", how="left")

    clinical, clinical_labels = load_clinical(root)
    if not clinical.empty:
        meta = runs.drop_duplicates("config_hash").set_index("config_hash")[["setup"]]
        clinical = clinical.join(meta, on="config_hash")
        clinical_labels = clinical_labels.join(meta, on="config_hash")
    return Results(runs=runs, curves=cat(curves), val_labels=cat(val_labels), test=cat(test),
                   test_labels=cat(test_labels), settings=settings, clinical=clinical, clinical_labels=clinical_labels)


def load_clinical(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every clinical report under ``root`` (a run's ``clinical/<eval>/``, a config folder's
    ``ensemble/clinical/<eval>/``): (one headline row per report, one row per report x group x label)."""
    rows, labels = [], []
    for card_path in sorted(Path(root).rglob("clinical/*/scorecard.json")):
        card = read_json(card_path)
        if not card:
            continue
        ids = {"run_id": card.get("run_id"), "name": card.get("name"), "config_hash": card.get("config_hash"),
               "hash8": (card.get("config_hash") or "")[:8], "seed": card.get("seed"),
               "ensemble": card.get("seed") == "ensemble", "members": card.get("members"), "eval": card.get("eval"),
               "eval_split": (card.get("eval") or "").rsplit("__", 1)[-1], "reported": card.get("reported"),
               "clinical_config": card.get("clinical_config_fingerprint")}
        rows.append({**ids, **card.get("headline", {}), "dir": card_path.parent.as_posix()})
        table = _read_csv(card_path.parent / "per_label.csv")
        gate = _read_csv(card_path.parent / "gate.csv")
        if table is not None:
            if gate is not None:
                table = table.merge(gate[["finding", "status"]], on="finding", how="left")
            for key, value in reversed(list(ids.items())):
                table.insert(0, key, value)
            labels.append(table)
    return pd.DataFrame(rows), (pd.concat(labels, ignore_index=True) if labels else pd.DataFrame())


def mean_std(frame: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Average ``metrics`` over the seeds of each group: columns ``<metric>`` (mean), ``<metric>_std``, ``seeds``."""
    grouped = frame.groupby(keys, dropna=False, sort=False)
    mean, std = grouped[metrics].mean(), grouped[metrics].std()
    out = mean.join(std.add_suffix("_std"))
    out.insert(0, "seeds", grouped.size())
    return out[["seeds"] + [c for m in metrics for c in (m, f"{m}_std")]]


def pm(table: pd.DataFrame, metrics: list[str], digits: int = 4) -> pd.DataFrame:
    """``0.8083 ± 0.0005`` strings for display, from a :func:`mean_std` table."""
    out = table[["seeds"]].copy()
    for m in metrics:
        out[m] = [f"{a:.{digits}f} ± {b:.{digits}f}" if pd.notna(b) else f"{a:.{digits}f}"
                  for a, b in zip(table[m], table[f"{m}_std"])]
    return out
