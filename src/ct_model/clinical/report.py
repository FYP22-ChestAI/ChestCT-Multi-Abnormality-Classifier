"""Write the clinical report of one evaluation -- of one run, or of a seed ensemble.

It reads only saved predictions, never a model, so any phase that writes the standard prediction files
(ct_model.training.trainer.predictions_frame) is scored the same way:

    <run>/predictions_val.csv                     VAL: chooses calibration + operating points
    <run>/eval/<source>__<run>__<split>/predictions.csv     the split being scored

``target`` is a run folder (``<name>/<hash8>/seed<k>``) or a CONFIG folder (``<name>/<hash8>``): then the
probabilities of every seed that has this evaluation are averaged (a seed ensemble), and with
``per_seed`` each seed also gets its own report.

    <run>/clinical/<eval>/  or  <name>/<hash8>/ensemble/clinical/<eval>/
        scorecard.json            headline numbers + label status (the one file to read first)
        per_label.csv             every label x group (all, kernel class, scanner): ranking, operating points,
                                  calibration, net benefit
        summary.csv               per group: false positives / missed findings per scan, exact match, ...
        per_scan.csv              one row per volume
        paired_kernel.csv         per label: decision changes between sharp and soft reconstructions
        paired_scans.csv          one row per sharp / soft pair
        gate.csv                  per label: each gate rule and the status
        operating_points.json     calibration and thresholds, chosen on val
        predictions_clinical.csv  calibrated probabilities and the 3-state output per label
        predictions.csv, predictions_val.csv   (ensemble only) the averaged probabilities
    outputs/experiments/clinical.csv              one row per report (ledger; the newest row per run and eval counts)

A report is derived and deterministic, so running it again simply rewrites it.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from ..utils.log import say
from ..training.records import append_row, ensure_dir, now_utc, read_json, write_json
from . import scorecard as SC
from .config import DEFAULT_CLINICAL_CONFIG, ClinicalConfig, load_clinical_config
from .operating import PRESENT, choose_operating_points

CLINICAL_LEDGER = "clinical.csv"
ID_COLUMNS = ("volume_id", "patient_id", "scan_id", "split", "kernel_class", "manufacturer")
GROUP_COLUMNS = ("kernel_class", "manufacturer")


@dataclass
class ClinicalResult:
    dir: Path
    scorecard: dict
    per_label: pd.DataFrame
    gate: pd.DataFrame


# ------------------------------------------------------------------ finding the inputs
def is_run_dir(folder: Path) -> bool:
    return (folder / "run_info.json").is_file() and (folder / "config.yaml").is_file()


def seed_dirs(config_dir: Path) -> list[Path]:
    return sorted((d for d in config_dir.glob("seed*") if is_run_dir(d)), key=lambda d: int(d.name[4:] or 0))


def eval_name(run_dir: Path, split: str = "test", source: str | None = None, run: str | None = None) -> str:
    """``<source>__<run>__<split>``, the folder evaluate_mil.py writes; defaults from the run's config."""
    data = (yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}).get("data", {})
    return f"{source or data.get('source')}__{run or data.get('run')}__{split}"


def output_dir(target: Path, name: str) -> Path:
    return target / "clinical" / name if is_run_dir(target) else target / "ensemble" / "clinical" / name


def _labels(run_dir: Path) -> list[str]:
    import json
    return json.loads((run_dir / "labels.json").read_text(encoding="utf-8"))


def _arrays(frame: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    missing = [n for n in names if f"prob_{n}" not in frame or f"true_{n}" not in frame]
    if missing:
        raise ValueError(f"predictions lack prob_/true_ columns for {missing}")
    p = frame[[f"prob_{n}" for n in names]].to_numpy(dtype=np.float64)
    y = frame[[f"true_{n}" for n in names]].to_numpy(dtype=np.float64)
    return y, p


def _average(frames: list[pd.DataFrame], names: list[str], what: str) -> pd.DataFrame:
    """Mean probability over members, aligned on volume_id (every member must hold the same volumes)."""
    base = frames[0].set_index("volume_id")
    for k, f in enumerate(frames[1:], start=1):
        if set(f.volume_id) != set(base.index):
            raise ValueError(f"ensemble member {k} has other {what} volumes than member 0 -- not the same evaluation")
    out = base.copy()
    probs = [f"prob_{n}" for n in names]
    out[probs] = np.mean([f.set_index("volume_id").loc[base.index, probs].to_numpy() for f in frames], axis=0)
    return out.reset_index()


def _manifest_columns(run_dir: Path, eval_dir: Path | None, log: Callable[[str], None]) -> pd.DataFrame | None:
    """scan_id / manufacturer per volume from the evaluated run's manifest (older predictions lack them)."""
    try:
        from ct_preprocessing.config import load_config
        from ..data.volumes import load_run_manifest

        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
        info = read_json(eval_dir / "eval_info.json") if eval_dir else {}
        data = cfg.get("data", {})
        _, manifest = load_run_manifest(load_config(cfg.get("data_config", "configs/preprocessing.yaml")),
                                        info.get("source") or data.get("source"), info.get("run") or data.get("run"))
        cols = [c for c in ("volume_id", "scan_id", "manufacturer") if c in manifest.columns]
        return manifest[cols].drop_duplicates("volume_id")
    except Exception as exc:  # the report still works, only without the paired / scanner views
        log(f"  note: could not read the run manifest ({exc}); paired-kernel and scanner views may be missing")
        return None


def _with_ids(frame: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    if extra is not None:
        add = [c for c in extra.columns if c != "volume_id" and c not in frame.columns]
        if add:
            frame = frame.merge(extra[["volume_id", *add]], on="volume_id", how="left")
    return frame


# ------------------------------------------------------------------ the report
def score(val: pd.DataFrame, test: pd.DataFrame, names: list[str], cfg: ClinicalConfig, seed: int = 0,
          log: Callable[[str], None] = say) -> dict:
    """Everything the report holds, as tables and dicts (no files): operating points from ``val``,
    scored on ``test``."""
    y_val, p_val = _arrays(val, names)
    y, p_raw = _arrays(test, names)
    k_val = val["kernel_class"].astype(str).to_numpy() if "kernel_class" in val else None
    kernels = test["kernel_class"].astype(str).to_numpy() if "kernel_class" in test else None
    ops = choose_operating_points(y_val, p_val, names, cfg, k_val)
    p = ops.calibrate(p_raw, kernels)
    patients = test["patient_id"].to_numpy() if "patient_id" in test else np.arange(len(test))
    ids = test[[c for c in ID_COLUMNS if c in test.columns]].reset_index(drop=True)

    groups = [("all", "all", np.arange(len(test)))]
    for col in GROUP_COLUMNS:
        if col in test.columns:
            groups += [(col, str(g), idx) for g, idx in test.groupby(test[col].fillna("unknown")).indices.items()]
    scans = SC.per_scan(ids, y, p, ops)
    label_tables, summaries = [], []
    for k, (group_by, group, idx) in enumerate(groups, start=1):
        big = len(idx) >= cfg.min_group
        ci = big and group_by in cfg.ci_groups
        log(f"  [{k}/{len(groups)}] {group_by}={group}: {len(idx)} volumes" + (f", bootstrap {cfg.bootstrap}" if ci and cfg.bootstrap else ""))
        t = SC.per_label(y[idx], p[idx], p_raw[idx], ops, cfg, patients[idx], with_ci=ci, seed=seed)
        t.insert(0, "group", group)
        t.insert(0, "group_by", group_by)
        label_tables.append(t)
        s = SC.scan_summary(scans.iloc[idx])
        summaries.append({"group_by": group_by, "group": group, "too_small": not big,
                          "n_patients": int(len(np.unique(patients[idx]))), **s,
                          "macro_auroc": float(np.nanmean(t.auroc)), "macro_auprc": float(np.nanmean(t.auprc)),
                          "report_macro_sens": float(np.nanmean(t.report_sens)),
                          "report_macro_spec": float(np.nanmean(t.report_spec)),
                          "report_macro_ppv": float(np.nanmean(t.report_ppv)),
                          "maxf1_macro_sens": float(np.nanmean(t.maxf1_sens)),
                          "maxf1_macro_spec": float(np.nanmean(t.maxf1_spec)),
                          "macro_ece_raw": float(np.nanmean(t.ece_raw)), "macro_ece": float(np.nanmean(t.ece))})
    per_label = pd.concat(label_tables, ignore_index=True)
    summary = pd.DataFrame(summaries)

    paired_table, paired_scans, paired_summary = SC.paired(ids, p, ops)
    by_kernel = per_label[per_label.group_by == "kernel_class"].pivot(index="label", columns="group", values="report_sens")
    gap = (by_kernel["soft"] - by_kernel["sharp"]) if {"sharp", "soft"} <= set(by_kernel.columns) else None
    labels_all = per_label[per_label.group_by == "all"].reset_index(drop=True)
    gate = SC.gate(labels_all, gap, paired_table, cfg)

    clinical = ids.copy()
    states = ops.states(p)
    for c, n in enumerate(names):
        clinical[f"cal_{n}"] = p[:, c]
        clinical[f"state_{n}"] = pd.Categorical.from_codes(states[:, c], ["absent", "uncertain", "present"])
        clinical[f"true_{n}"] = y[:, c]
    return {"ops": ops, "per_label": per_label, "summary": summary, "per_scan": scans, "paired": paired_table,
            "paired_scans": paired_scans, "paired_summary": paired_summary, "gate": gate, "kernel_gap": gap,
            "predictions": clinical, "states": states}


def _headline(parts: dict, cfg: ClinicalConfig) -> dict:
    summary = parts["summary"].set_index(["group_by", "group"])
    overall = summary.loc[("all", "all")]
    gate = parts["gate"]
    head = {
        "n_volumes": int(overall.n_volumes), "n_patients": int(overall.n_patients), "n_normal": int(overall.n_normal),
        "macro_auroc": overall.macro_auroc, "macro_auprc": overall.macro_auprc,
        "labels_supported": int((gate.status == SC.SUPPORTED).sum()),
        "labels_review": int((gate.status == SC.REVIEW).sum()),
        "labels_not_supported": int((gate.status == SC.NOT_SUPPORTED).sum()),
        "true_per_scan": overall.true_per_scan,
    }
    for rule in ("report", "maxf1"):
        for k in ("pred_per_scan", "fp_per_scan", "fn_per_scan", "scans_with_fp", "exact_match", "normal_flag_rate"):
            head[f"{rule}_{k}"] = overall[f"{rule}_{k}"]
        head[f"{rule}_macro_sens"], head[f"{rule}_macro_spec"] = overall[f"{rule}_macro_sens"], overall[f"{rule}_macro_spec"]
    head["report_macro_ppv"] = overall.report_macro_ppv
    head["uncertain_per_scan"], head["scans_with_uncertain"] = overall.uncertain_per_scan, overall.scans_with_uncertain
    head["macro_ece_raw"], head["macro_ece"] = overall.macro_ece_raw, overall.macro_ece
    head.update(parts["paired_summary"])
    for kc in SC.PAIR_KERNELS:
        if ("kernel_class", kc) in summary.index:
            row = summary.loc[("kernel_class", kc)]
            head[f"report_macro_sens[{kc}]"] = row.report_macro_sens
            head[f"report_fp_per_scan[{kc}]"] = row.report_fp_per_scan
            head[f"macro_ece_raw[{kc}]"], head[f"macro_ece[{kc}]"] = row.macro_ece_raw, row.macro_ece
            head[f"macro_auroc[{kc}]"] = row.macro_auroc
    if parts["kernel_gap"] is not None:
        head["kernel_sens_gap_mean_abs"] = float(np.nanmean(np.abs(parts["kernel_gap"])))
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in head.items()}


def _write(out: Path, parts: dict, cfg: ClinicalConfig, info: dict) -> dict:
    if out.exists():
        shutil.rmtree(out)
    ensure_dir(out)
    csv = {"index": False, "lineterminator": "\n"}
    parts["per_label"].to_csv(out / "per_label.csv", **csv)
    parts["summary"].to_csv(out / "summary.csv", **csv)
    parts["per_scan"].to_csv(out / "per_scan.csv", **csv)
    parts["paired"].to_csv(out / "paired_kernel.csv", **csv)
    parts["paired_scans"].to_csv(out / "paired_scans.csv", **csv)
    parts["gate"].to_csv(out / "gate.csv", **csv)
    parts["predictions"].to_csv(out / "predictions_clinical.csv", **csv)
    op = cfg.operating_points
    write_json(out / "operating_points.json", parts["ops"].to_dict(op.report_specificity, op.ruleout_sensitivity))
    gate = parts["gate"]
    card = {**info, "reported": now_utc(), "clinical_config": cfg.to_dict(), "clinical_config_fingerprint": cfg.fingerprint(),
            "headline": _headline(parts, cfg),
            "labels": {s: gate.label[gate.status == s].tolist() for s in (SC.SUPPORTED, SC.REVIEW, SC.NOT_SUPPORTED)}}
    write_json(out / "scorecard.json", card)
    return card


def _report_one(member_dirs: list[Path], out: Path, name: str, cfg: ClinicalConfig, info: dict,
                output_root: Path | None, log: Callable[[str], None]) -> ClinicalResult:
    names = _labels(member_dirs[0])
    for d in member_dirs[1:]:
        if _labels(d) != names:
            raise ValueError(f"{d} has other labels than {member_dirs[0]} -- cannot ensemble them")
    vals, tests = [], []
    for d in member_dirs:
        ev = d / "eval" / name / "predictions.csv"
        if not (d / "predictions_val.csv").is_file():
            raise FileNotFoundError(f"{d / 'predictions_val.csv'} is missing -- is this a finished training run?")
        if not ev.is_file():
            raise FileNotFoundError(f"{ev} is missing -- run scripts/model/evaluate_mil.py --experiment-dir {d} first")
        vals.append(pd.read_csv(d / "predictions_val.csv"))
        tests.append(pd.read_csv(ev))
    val = vals[0] if len(vals) == 1 else _average(vals, names, "val")
    test = tests[0] if len(tests) == 1 else _average(tests, names, name)
    extra = _manifest_columns(member_dirs[0], member_dirs[0] / "eval" / name, log)
    val, test = _with_ids(val, extra), _with_ids(test, extra)

    seed = int(yaml.safe_load((member_dirs[0] / "config.yaml").read_text(encoding="utf-8")).get("seed", 0) or 0)
    parts = score(val, test, names, cfg, seed=seed, log=log)
    card = _write(out, parts, cfg, info)
    if len(member_dirs) > 1:
        csv = {"index": False, "lineterminator": "\n"}
        val.to_csv(out / "predictions_val.csv", **csv)
        test.to_csv(out / "predictions.csv", **csv)

    h = card["headline"]
    if output_root is not None:
        append_row(output_root / CLINICAL_LEDGER, {
            "run_id": info["run_id"], "name": info.get("name"), "config_hash": info.get("config_hash"),
            "seed": info.get("seed"), "members": info["members"], "eval": name, "eval_split": name.rsplit("__", 1)[-1],
            "reported": card["reported"], "clinical_config": card["clinical_config_fingerprint"],
            **{k: v for k, v in h.items() if not isinstance(v, (list, dict))}, "dir": out.as_posix(),
        })
    log(f"  {info['run_id']} on {name}: labels supported {h['labels_supported']} / review {h['labels_review']} / "
        f"not supported {h['labels_not_supported']}; report rule: {h['report_fp_per_scan']:.2f} false positives and "
        f"{h['report_fn_per_scan']:.2f} missed per scan, normal scans flagged {h['report_normal_flag_rate']:.1%}"
        + (f"; paired sharp/soft scans with a changed decision {h['paired_scans_with_flip']:.1%}" if h.get("n_pairs") else ""))
    log(f"  written to {out}")
    return ClinicalResult(out, card, parts["per_label"], parts["gate"])


def _output_root(run_dir: Path) -> Path | None:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    return Path(cfg.get("output_dir", "outputs/experiments"))


def _run_info(run_dir: Path) -> dict:
    info = read_json(run_dir / "run_info.json")
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}
    return {"run_id": info.get("run_id", run_dir.as_posix()), "name": cfg.get("name"),
            "config_hash": info.get("config_hash"), "seed": cfg.get("seed")}


def clinical_report(target: str | Path, *, split: str = "test", source: str | None = None, run: str | None = None,
                    config: str | Path | ClinicalConfig = DEFAULT_CLINICAL_CONFIG, per_seed: bool = False,
                    ledger: bool = True, log: Callable[[str], None] = say) -> list[ClinicalResult]:
    """Report one run folder, or a config folder (seed ensemble; ``per_seed`` adds every seed's own report).
    Returns the results written, the ensemble last."""
    target = Path(target)
    cfg = config if isinstance(config, ClinicalConfig) else load_clinical_config(config)
    if is_run_dir(target):
        name = eval_name(target, split, source, run)
        info = {**_run_info(target), "members": 1, "eval": name}
        root = _output_root(target) if ledger else None
        return [_report_one([target], output_dir(target, name), name, cfg, info, root, log)]

    seeds = seed_dirs(target)
    if not seeds:
        raise FileNotFoundError(f"{target} is neither a run folder (config.yaml + run_info.json) nor a config folder "
                                "holding seed<k>/ runs")
    name = eval_name(seeds[0], split, source, run)
    members = [d for d in seeds if (d / "eval" / name / "predictions.csv").is_file()
               and read_json(d / "run_info.json").get("status") == "finished"]
    if not members:
        raise FileNotFoundError(f"no seed under {target} has the evaluation {name} -- run evaluate_mil.py first")
    root = _output_root(members[0]) if ledger else None
    results = []
    if per_seed:
        for d in members:
            info = {**_run_info(d), "members": 1, "eval": name}
            results.append(_report_one([d], output_dir(d, name), name, cfg, info, root, log))
    first = _run_info(members[0])
    rid = first["run_id"].rsplit("/", 1)[0] + "/ensemble" if "/" in first["run_id"] else f"{target.as_posix()}/ensemble"
    info = {"run_id": rid, "name": first["name"], "config_hash": first["config_hash"], "seed": "ensemble",
            "members": len(members), "member_runs": [_run_info(d)["run_id"] for d in members], "eval": name}
    log(f"ensemble of {len(members)} seed(s): {', '.join(d.name for d in members)}")
    results.append(_report_one(members, output_dir(target, name), name, cfg, info, root, log))
    return results
