"""Export what is under outputs/ to an MLflow tracking server (e.g. DagsHub) -- after the fact, idempotently.

The run folders stay the source of truth; MLflow is a shareable VIEW of them. Every export walks the folders
again and only adds what the server does not have yet, so it can run after every phase (or in the job chain).

    MLflow experiment                 MLflow run (name)              from
    -----------------                 ------------------              ----
    <experiment name>                 <hash8>/seed<k>                 outputs/experiments/<name>/<hash8>/seed<k>/
    <experiment name>                 <hash8>/ensemble                outputs/experiments/<name>/<hash8>/ensemble/clinical/
    adapt/<adapt config name>         <hash8>/seed<k>                 outputs/adapters/<name>/<hash8>/seed<k>/

* params: the resolved config, flattened (``aggregator.branches`` ...), plus the data run, embedding store, seed;
* metrics: every epoch's curve (step = epoch), the final val metrics, each test evaluation (overall, per kernel,
  per scanner, per label: ``test.macro_auroc.sharp``, ``test.label.cardiomegaly.auroc``) and the clinical headline
  (``clinical.test.report_fp_per_scan`` ...); an evaluation on another data run is keyed ``<source>.<run>.<split>``;
* tags: ``ct.run_id`` (how an exported run is recognised next time), git commit, status, clinical label status;
* artifacts: configs, metrics tables, evaluation summaries, clinical reports, and the trained weights (``best.pt``;
  LoRA ``adapter.safetensors`` + ``head.pt``). NOT exported: the HU cache, embedding stores, resume checkpoints
  (``last.pt``), and per-volume prediction files -- they hold CT-RATE patient-level data and labels (CC BY-NC-SA
  terms) -- unless ``with_predictions`` is set.

A run already on the server is skipped; ``refresh`` deletes and re-exports it (e.g. after a new evaluation).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from ..training.records import read_json
from ..training.results import find_run_dirs, flatten
from ..utils.log import say

RUN_TAG = "ct.run_id"
_MAX_PARAM = 500
_PRIVATE_FILES = ("predictions.csv", "predictions_val.csv", "predictions_clinical.csv", "per_scan.csv", "paired_scans.csv")


@dataclass
class ExportSummary:
    exported: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _key(*parts) -> str:
    """MLflow keys allow letters, digits, _ - . / and spaces; anything else becomes _."""
    raw = ".".join(str(p) for p in parts if p not in (None, ""))
    return "".join(c if c.isalnum() or c in "_-./ " else "_" for c in raw)


def eval_key(eval_name: str, cfg: dict) -> str:
    """``test`` for an evaluation on the run the model was trained on, else ``<source>.<run>.<split>``
    (another dataset, e.g. NHRD), so the usual case gets short metric names (``test.macro_auroc``)."""
    source, run, split = (eval_name.split("__") + ["", "", ""])[:3]
    data = (cfg or {}).get("data") or {}
    return split if (source, run) == (data.get("source"), data.get("run")) else _key(source, run, split)


def _params(cfg: dict, extra: dict) -> dict:
    out = {k: v for k, v in flatten(cfg).items() if k not in ("name", "output_dir")}
    out.update(extra)
    return {_key(k): ("null" if v is None else str(v))[:_MAX_PARAM] for k, v in out.items()}


class Exporter:
    def __init__(self, tracking_uri: str | None = None, with_predictions: bool = False, with_weights: bool = True,
                 refresh: bool = False, dry_run: bool = False, log: Callable[[str], None] = say):
        import mlflow
        from mlflow.tracking import MlflowClient

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.uri = mlflow.get_tracking_uri()
        self.with_predictions, self.with_weights = with_predictions, with_weights
        self.refresh, self.dry_run, self.log = refresh, dry_run, log
        self._experiments: dict[str, str] = {}

    # ------------------------------------------------------------------ server side
    def experiment_id(self, name: str) -> str:
        if name not in self._experiments:
            exp = self.client.get_experiment_by_name(name)
            self._experiments[name] = exp.experiment_id if exp else self.client.create_experiment(name)
        return self._experiments[name]

    def existing(self, experiment: str, run_id: str):
        exp = self.client.get_experiment_by_name(experiment)
        if exp is None:
            return None
        found = self.client.search_runs([exp.experiment_id], filter_string=f"tags.`{RUN_TAG}` = '{run_id}'", max_results=1)
        return found[0] if found else None

    def _begin(self, experiment: str, run_id: str, run_name: str, summary: ExportSummary):
        old = self.existing(experiment, run_id) if not self.dry_run else None
        if old is not None and not self.refresh:
            summary.skipped.append(run_id)
            return None
        if self.dry_run:
            self.log(f"  would export {experiment} :: {run_name}")
            summary.exported.append(run_id)
            return None
        if old is not None:
            self.client.delete_run(old.info.run_id)
        run = self.client.create_run(self.experiment_id(experiment), run_name=run_name, tags={RUN_TAG: run_id})
        return run.info.run_id

    def _log(self, mlrun: str, params: dict, metrics: list[tuple[str, float, int]], tags: dict) -> None:
        import time

        from mlflow.entities import Metric, Param, RunTag

        now = int(time.time() * 1000)
        p = [Param(k, v) for k, v in params.items()]
        m = [Metric(_key(k), v, now, step) for k, v, step in metrics if v is not None]
        t = [RunTag(_key(k), str(v)[:5000]) for k, v in tags.items() if v is not None]
        for i in range(0, len(p), 100):  # the REST API caps a batch at 100 params / tags and 1000 metrics
            self.client.log_batch(mlrun, params=p[i:i + 100])
        for i in range(0, len(t), 100):
            self.client.log_batch(mlrun, tags=t[i:i + 100])
        for i in range(0, len(m), 900):
            self.client.log_batch(mlrun, metrics=m[i:i + 900])

    def _artifacts(self, mlrun: str, folder: Path, files: list[str], dest: str | None = None) -> None:
        for name in files:
            f = folder / name
            if f.is_file() and (self.with_predictions or f.name not in _PRIVATE_FILES):
                self.client.log_artifact(mlrun, str(f), dest)

    def _clinical(self, mlrun: str, folder: Path, metrics: list, tags: dict, dest: str, key: str) -> None:
        card = read_json(folder / "scorecard.json")
        for k, v in (card.get("headline") or {}).items():
            metrics.append((_key("clinical", key, k.replace("[", ".").replace("]", "")), _num(v), 0))
        for status, names in (card.get("labels") or {}).items():
            tags[f"clinical.{key}.{status}"] = ", ".join(names) or "-"
        tags["clinical.config"] = card.get("clinical_config_fingerprint")
        self._artifacts(mlrun, folder, ["scorecard.json", "per_label.csv", "summary.csv", "gate.csv", "paired_kernel.csv",
                                        "operating_points.json", "predictions_clinical.csv", "per_scan.csv"], dest)

    # ------------------------------------------------------------------ one training run
    def export_run(self, folder: Path, summary: ExportSummary) -> None:
        info = read_json(folder / "run_info.json")
        cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8")) or {}
        rid = info.get("run_id") or folder.as_posix()
        if info.get("status") != "finished":
            summary.skipped.append(f"{rid} (status {info.get('status')})")
            return
        name, hash8 = cfg.get("name") or folder.parent.parent.name, folder.parent.name
        mlrun = self._begin(name, rid, f"{hash8}/{folder.name}", summary)
        if mlrun is None:
            return
        data = info.get("data", {})
        params = _params(cfg, {"config_hash": info.get("config_hash"), "data.run_resolved": data.get("run"),
                               "embedding_store": Path(str(data.get("embedding_store", ""))).name,
                               "train_volumes": data.get("train_volumes"), "val_volumes": data.get("val_volumes")})
        metrics: list[tuple[str, float | None, int]] = []
        curves = folder / "metrics.csv"
        if curves.is_file():
            for _, row in pd.read_csv(curves).iterrows():
                for col, v in row.items():
                    if col not in ("epoch", "improved", "step") and not col.startswith("val_auroc_"):
                        metrics.append((col, _num(v), int(row["epoch"])))
        for k, v in read_json(folder / "val_metrics.json").items():
            metrics.append((k if k.startswith("val_") else f"val_{k}", _num(v), 0))
        tags = {"ct.status": info.get("status"), "ct.stopped": info.get("stopped"), "ct.best_epoch": info.get("best_epoch"),
                "ct.config_hash": info.get("config_hash"), "ct.source_dir": folder.as_posix(),
                "mlflow.source.git.commit": (info.get("git") or {}).get("commit"), "ct.git_dirty": (info.get("git") or {}).get("dirty"),
                "ct.gpu": (info.get("environment") or {}).get("gpu")}
        self._artifacts(mlrun, folder, ["config.yaml", "run_info.json", "labels.json", "store_record.json", "metrics.csv",
                                        "val_per_label.csv", "thresholds.json", "val_metrics.json", "predictions_val.csv"])
        if self.with_weights:
            self._artifacts(mlrun, folder / "checkpoints", ["best.pt"], "weights")
        for ev in sorted((folder / "eval").glob("*")) if (folder / "eval").is_dir() else []:
            split = eval_key(ev.name, cfg)
            s = pd.read_csv(ev / "metrics_summary.csv") if (ev / "metrics_summary.csv").is_file() else pd.DataFrame()
            for _, row in s.iterrows():
                group = "" if row["group_by"] == "all" else str(row["group"])
                for col in ("macro_auroc", "macro_auroc_ci_low", "macro_auroc_ci_high", "macro_auprc", "micro_auroc",
                            "macro_f1", "macro_sensitivity", "macro_specificity", "macro_ppv"):
                    if col in row:
                        metrics.append((_key(split, col, group), _num(row[col]), 0))
            pl = pd.read_csv(ev / "metrics_per_label.csv") if (ev / "metrics_per_label.csv").is_file() else pd.DataFrame()
            for _, row in pl[pl.group_by == "all"].iterrows() if len(pl) else []:
                for col in ("auroc", "auprc"):
                    metrics.append((_key(split, "label", row["label"], col), _num(row[col]), 0))
            self._artifacts(mlrun, ev, ["metrics_summary.csv", "metrics_per_label.csv", "eval_info.json", "predictions.csv"],
                            f"eval/{ev.name}")
            if (folder / "clinical" / ev.name / "scorecard.json").is_file():
                self._clinical(mlrun, folder / "clinical" / ev.name, metrics, tags, f"clinical/{ev.name}", split)
        self._log(mlrun, params, metrics, tags)
        self.client.set_terminated(mlrun)
        summary.exported.append(rid)

    # ------------------------------------------------------------------ a seed ensemble
    def export_ensemble(self, config_dir: Path, summary: ExportSummary) -> None:
        reports = sorted((config_dir / "ensemble" / "clinical").glob("*/scorecard.json"))
        if not reports:
            return
        first = read_json(reports[0])
        name = first.get("name") or config_dir.parent.name
        rid = first.get("run_id") or f"{config_dir.as_posix()}/ensemble"
        mlrun = self._begin(name, rid, f"{config_dir.name}/ensemble", summary)
        if mlrun is None:
            return
        seed_cfg = next(iter(sorted(config_dir.glob("seed*/config.yaml"))), None)
        cfg = yaml.safe_load(seed_cfg.read_text(encoding="utf-8")) if seed_cfg else {}
        cfg.pop("seed", None)
        metrics, tags = [], {"ct.ensemble": True, "ct.members": first.get("members"), "ct.config_hash": first.get("config_hash"),
                             "ct.member_runs": ", ".join(first.get("member_runs") or []), "ct.source_dir": config_dir.as_posix()}
        for card in reports:
            self._clinical(mlrun, card.parent, metrics, tags, f"clinical/{card.parent.name}", eval_key(card.parent.name, cfg))
        self._log(mlrun, _params(cfg, {"config_hash": first.get("config_hash"), "seeds": first.get("members")}), metrics, tags)
        self.client.set_terminated(mlrun)
        summary.exported.append(rid)

    # ------------------------------------------------------------------ a LoRA adaptation run
    def export_adapter(self, folder: Path, summary: ExportSummary) -> None:
        info = read_json(folder / "run_info.json")
        if info.get("status") != "finished":
            summary.skipped.append(f"{info.get('run_id', folder)} (status {info.get('status')})")
            return
        cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8")) or {}
        rid = f"adapt/{info.get('run_id')}"
        mlrun = self._begin(f"adapt/{cfg.get('name')}", rid, f"{folder.parent.name}/{folder.name}", summary)
        if mlrun is None:
            return
        metrics = []
        if (folder / "metrics.csv").is_file():
            for _, row in pd.read_csv(folder / "metrics.csv").iterrows():
                for col, v in row.items():
                    if col not in ("epoch", "improved"):
                        metrics.append((col.replace("[", ".").replace("]", ""), _num(v), int(row["epoch"])))
        tags = {"ct.config_hash": info.get("config_hash"), "ct.best_epoch": info.get("best_epoch"),
                "ct.encoder_yaml": info.get("encoder_yaml"), "ct.experiment_yaml": info.get("experiment_yaml"),
                "ct.init_from": cfg.get("init_from"), "ct.source_dir": folder.as_posix(),
                "mlflow.source.git.commit": (info.get("git") or {}).get("commit")}
        self._artifacts(mlrun, folder, ["config.yaml", "run_info.json", "metrics.csv", "encoder.yaml", "experiment.yaml"])
        if self.with_weights:
            self._artifacts(mlrun, folder, ["adapter.safetensors", "head.pt"], "weights")
        self._log(mlrun, _params(cfg, {"config_hash": info.get("config_hash"),
                                       "lora_parameters": info.get("lora_parameters")}), metrics, tags)
        self.client.set_terminated(mlrun)
        summary.exported.append(rid)


def export_outputs(experiments_dir: str | Path = "outputs/experiments", adapters_dir: str | Path | None = "outputs/adapters",
                   *, name: str | None = None, **kwargs) -> ExportSummary:
    """Export every finished run, seed ensemble and LoRA adaptation under the two folders (``name``: only that
    experiment / adapt config)."""
    ex = Exporter(**kwargs)
    ex.log(f"MLflow tracking server: {ex.uri}")
    summary = ExportSummary()
    folders = [f for f in find_run_dirs(experiments_dir) if name in (None, f.parent.parent.name)]
    for f in folders:
        try:
            ex.export_run(f, summary)
        except Exception as exc:  # one broken folder never stops the export
            summary.failed[f.as_posix()] = repr(exc)
    for config_dir in sorted({f.parent for f in folders}):
        try:
            ex.export_ensemble(config_dir, summary)
        except Exception as exc:
            summary.failed[config_dir.as_posix()] = repr(exc)
    if adapters_dir and Path(adapters_dir).is_dir():
        for info in sorted(Path(adapters_dir).glob("*/*/seed*/run_info.json")):
            if name in (None, info.parent.parent.parent.name):
                try:
                    ex.export_adapter(info.parent, summary)
                except Exception as exc:
                    summary.failed[info.parent.as_posix()] = repr(exc)
    ex.log(f"exported {len(summary.exported)}, skipped {len(summary.skipped)} (already there / unfinished), "
           f"failed {len(summary.failed)}")
    for k, v in summary.failed.items():
        ex.log(f"  FAILED {k}: {v}")
    return summary
