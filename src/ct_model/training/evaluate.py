"""Evaluate a FINISHED training run on a split -- normally test, once, after all choices were made on val.

Uses the best checkpoint and the thresholds chosen on val (never re-tuned here). Reports every label
and the macro / micro summaries:
  * overall, with a patient-level bootstrap 95 % CI of macro AUROC and macro AUPRC;
  * per kernel class (sharp / soft / other) -- test holds both kernels of each scan, and training saw
    only sharp, so sharp vs soft is the cross-kernel result;
  * per scanner manufacturer -- 10 of the 18 CT-RATE labels depend on the scanner.
A group with fewer than ``min_group`` volumes is still reported, flagged ``too_small``, and gets no CI.

It can also score another run of the same labels (a soft-kernel subset, NHRD) without retraining,
as long as its volumes are in the same embedding store.

    <run dir>/eval/<source>__<run>__<split>/
        predictions.csv  metrics_summary.csv  metrics_per_label.csv  eval_info.json
    outputs/experiments/evaluations.csv      one row per evaluation (ledger)
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..utils.device import resolve_device
from ..utils.log import say
from . import metrics as M
from .data import open_data
from .records import EVAL_LEDGER, append_row, ensure_dir, now_utc, write_json
from .trainer import load_trained, predict, predictions_frame

GROUP_COLUMNS = ("kernel_class", "manufacturer")


class AlreadyEvaluated(RuntimeError):
    """This run was already evaluated on this split; results are never silently redone."""


@dataclass
class EvalResult:
    dir: Path
    summary: pd.DataFrame
    per_label: pd.DataFrame


def _group_rows(y, p, patients, names, thresholds, group_by, group, n_boot, seed, min_group):
    s = M.summary(y, p, thresholds)
    row = {"group_by": group_by, "group": group, "n_patients": int(len(np.unique(patients))),
           "too_small": bool(len(y) < min_group), **s}
    if n_boot and len(y) >= min_group:
        row["macro_auroc_ci_low"], row["macro_auroc_ci_high"] = M.bootstrap_ci(y, p, patients, M.macro_auroc, n_boot, seed)
        row["macro_auprc_ci_low"], row["macro_auprc_ci_high"] = M.bootstrap_ci(y, p, patients, M.macro_auprc, n_boot, seed)
    table = M.per_label_table(y, p, names, thresholds)
    table.insert(0, "group", group)
    table.insert(0, "group_by", group_by)
    return row, table


def evaluate_experiment(exp_dir: str | Path, *, split: str = "test", source: str | None = None, run: str | None = None,
                        kernel_classes: tuple[str, ...] | None = None, n_boot: int = 1000, min_group: int = 100,
                        device: str = "auto", overwrite: bool = False, log: Callable[[str], None] = say) -> EvalResult:
    dev = resolve_device(device)
    trained = load_trained(exp_dir, device=dev)
    cfg, names = trained.cfg, trained.label_names
    data = open_data(cfg, source=source, run=run, labels=names)
    out = Path(exp_dir) / "eval" / f"{data.run.source}__{data.run.name}__{split}"
    if out.exists():
        if not overwrite:
            raise AlreadyEvaluated(f"{out} exists -- this evaluation was already done (pass --overwrite to redo it)")
        shutil.rmtree(out)
    ensure_dir(out)

    dataset = data.dataset(split, kernel_classes, preload=True)
    log(f"evaluating {trained.run_info['run_id']} on {data.run.source}/{data.run.name} {split}: {len(dataset)} volumes")
    probs, _ = predict(trained.model, dataset, dev)
    y = dataset.labels()
    preds = predictions_frame(dataset, probs, names)
    extra = [c for c in GROUP_COLUMNS if c in data.manifest.columns and c != "kernel_class"]
    if extra:
        preds = preds.merge(data.manifest[["volume_id", *extra]].drop_duplicates("volume_id"), on="volume_id", how="left")
    preds.to_csv(out / "predictions.csv", index=False, lineterminator="\n")

    patients = preds["patient_id"].to_numpy()
    groups = [("all", "all", np.arange(len(preds)))]
    for col in GROUP_COLUMNS:
        if col in preds.columns:
            groups += [(col, str(g), idx) for g, idx in preds.groupby(preds[col].fillna("unknown")).indices.items()]
    rows, tables = [], []
    for k, (group_by, group, idx) in enumerate(groups, start=1):
        ci = f", bootstrap {n_boot} x 2 CIs" if n_boot and len(idx) >= min_group else ""
        log(f"  [{k}/{len(groups)}] {group_by}={group}: {len(idx)} volumes{ci}")
        row, table = _group_rows(y[idx], probs[idx], patients[idx], names, trained.thresholds, group_by, group,
                                 n_boot, cfg.seed, min_group)
        rows.append(row)
        tables.append(table)
    summary = pd.DataFrame(rows)
    per_label = pd.concat(tables, ignore_index=True)
    summary.to_csv(out / "metrics_summary.csv", index=False, lineterminator="\n")
    per_label.to_csv(out / "metrics_per_label.csv", index=False, lineterminator="\n")
    write_json(out / "eval_info.json", {
        "run_id": trained.run_info["run_id"], "config_hash": trained.run_info["config_hash"], "evaluated": now_utc(),
        "source": data.run.source, "run": data.run.name, "split": split, "kernel_classes": kernel_classes,
        "manifest_sha256": data.manifest_sha256, "n_volumes": len(dataset), "n_boot": n_boot,
        "thresholds": "max-F1 per label, chosen on val at training time", "checkpoint": "best",
    })

    overall = summary.iloc[0]
    ledger_row = {
        "run_id": trained.run_info["run_id"], "name": cfg.name, "config_hash": trained.run_info["config_hash"],
        "seed": cfg.seed, "evaluated": now_utc(), "eval_source": data.run.source, "eval_run": data.run.name,
        "eval_split": split, "n_volumes": len(dataset), "macro_auroc": overall["macro_auroc"],
        "macro_auroc_ci_low": overall.get("macro_auroc_ci_low"), "macro_auroc_ci_high": overall.get("macro_auroc_ci_high"),
        "macro_auprc": overall["macro_auprc"], "micro_auroc": overall["micro_auroc"], "macro_f1": overall["macro_f1"],
        "dir": str(out),
    }
    for _, r in summary[summary.group_by == "kernel_class"].iterrows():
        ledger_row[f"macro_auroc[{r['group']}]"] = r["macro_auroc"]
    append_row(Path(cfg.output_dir) / EVAL_LEDGER, ledger_row)
    log(f"macro AUROC {overall['macro_auroc']:.4f}"
        + (f" (95% CI {overall['macro_auroc_ci_low']:.4f}-{overall['macro_auroc_ci_high']:.4f})" if "macro_auroc_ci_low" in overall and pd.notna(overall.get("macro_auroc_ci_low")) else "")
        + f", macro AUPRC {overall['macro_auprc']:.4f}; results in {out}")
    return EvalResult(out, summary, per_label)
