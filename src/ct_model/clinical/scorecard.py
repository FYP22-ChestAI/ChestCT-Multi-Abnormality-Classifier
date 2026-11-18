"""The clinical scorecard: what a reader of the output would experience, not only how well it ranks.

Every function takes plain arrays: ``y`` (N, C) true labels with NaN = unknown, ``p`` (N, C) CALIBRATED
probabilities, and the operating points chosen on VAL (ct_model.clinical.operating).

* per label   AUROC (+ patient-bootstrap CI), AUPRC; sensitivity / specificity / PPV / NPV at the report
              threshold; sensitivity / NPV at the rule-out threshold and the uncertain rate between them;
              the legacy max-F1 point; sensitivity at fixed specificities; ECE (raw and calibrated), Brier;
              decision-curve net benefit.
* per scan    for the report rule and the legacy max-F1 rule: labels predicted, false positives, missed
              findings, exact match, normal scans flagged; uncertain labels per scan.
* paired      the same scan reconstructed with both kernels (patient_id + scan_id, one sharp and one soft
              volume): how often a decision changes, how often it is a HARD contradiction ("present" on one
              kernel, "absent" on the other), and how far the probability moves.
* gate        per label: supported / review / not_supported (rules in configs/model/clinical.yaml).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..training import metrics as M
from .config import ClinicalConfig
from .operating import ABSENT, PRESENT, UNCERTAIN, OperatingPoints

SUPPORTED, REVIEW, NOT_SUPPORTED = "supported", "review", "not_supported"
PAIR_KERNELS = ("sharp", "soft")


# ------------------------------------------------------------------ building blocks
def _div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b > 0)


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Per label tp / fp / fn / tn over the volumes whose label is known."""
    known = ~np.isnan(y)
    pos, neg = known & (y == 1), known & (y == 0)
    return {"tp": (pred & pos).sum(0), "fp": (pred & neg).sum(0), "fn": (~pred & pos).sum(0), "tn": (~pred & neg).sum(0)}


def rates(y: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    c = confusion(y, pred)
    return {"sens": _div(c["tp"], c["tp"] + c["fn"]), "spec": _div(c["tn"], c["tn"] + c["fp"]),
            "ppv": _div(c["tp"], c["tp"] + c["fp"]), "npv": _div(c["tn"], c["tn"] + c["fn"])}


def ece_per_label(y: np.ndarray, p: np.ndarray, bins: int = 10) -> np.ndarray:
    """Expected calibration error per label, equal-width bins."""
    out = np.full(y.shape[1], np.nan)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for c in range(y.shape[1]):
        t, q = M._known(y, p, c)
        if t.size == 0:
            continue
        idx = np.clip(np.digitize(q, edges[1:-1]), 0, bins - 1)
        err = 0.0
        for b in range(bins):
            m = idx == b
            if m.any():
                err += m.mean() * abs(t[m].mean() - q[m].mean())
        out[c] = err
    return out


def brier_per_label(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        t, q = M._known(y, p, c)
        if t.size:
            out[c] = float(np.mean((q - t) ** 2))
    return out


def net_benefit(y: np.ndarray, p: np.ndarray, pt: float) -> tuple[np.ndarray, np.ndarray]:
    """(model, treat-all) net benefit per label at threshold probability ``pt`` (decision curve analysis)."""
    w = pt / (1 - pt)
    model, treat_all = np.full(y.shape[1], np.nan), np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        t, q = M._known(y, p, c)
        if t.size:
            pred = q >= pt
            model[c] = ((pred & (t == 1)).sum() - w * (pred & (t == 0)).sum()) / t.size
            treat_all[c] = t.mean() - w * (1 - t.mean())
    return model, treat_all


def auroc_columns(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """AUROC of every label at once (rank formula, ties averaged); NaN where a label has one class.
    Same values as roc_auc_score, fast enough for bootstrapping."""
    if np.isnan(y).any():
        return M.auroc_per_label(y, p)
    ranks = rankdata(p, axis=0)
    n_pos = y.sum(0)
    n_neg = y.shape[0] - n_pos
    u = (ranks * y).sum(0) - n_pos * (n_pos + 1) / 2
    return _div(u, n_pos * n_neg)


def auroc_ci_per_label(y: np.ndarray, p: np.ndarray, patients, n_boot: int, seed: int = 0,
                       alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Per-label percentile CI of AUROC over ``n_boot`` resamples of PATIENTS (as metrics.bootstrap_ci)."""
    patients = np.asarray(patients)
    _, inverse = np.unique(patients, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    rng = np.random.default_rng(seed)
    values = np.empty((n_boot, y.shape[1]))
    for b in range(n_boot):
        idx = np.concatenate([rows_of[k] for k in rng.integers(0, len(rows_of), size=len(rows_of))])
        values[b] = auroc_columns(y[idx], p[idx])
    with np.errstate(all="ignore"):
        lo = np.nanquantile(values, alpha / 2, axis=0)
        hi = np.nanquantile(values, 1 - alpha / 2, axis=0)
    return lo, hi


# ------------------------------------------------------------------ per label
def per_label(y: np.ndarray, p: np.ndarray, p_raw: np.ndarray, ops: OperatingPoints, cfg: ClinicalConfig,
              patients=None, with_ci: bool = False, seed: int = 0) -> pd.DataFrame:
    known = ~np.isnan(y)
    states = ops.states(p)
    t = pd.DataFrame({"label": ops.label_names, "n": known.sum(0), "n_pos": np.nansum(y, axis=0).astype(int)})
    t["prevalence"] = _div(t.n_pos, t.n)
    t["auroc"] = auroc_columns(y, p)
    if with_ci and cfg.bootstrap and patients is not None:
        t["auroc_ci_low"], t["auroc_ci_high"] = auroc_ci_per_label(y, p, patients, cfg.bootstrap, seed)
    else:
        t["auroc_ci_low"] = t["auroc_ci_high"] = np.nan
    t["auprc"] = M.auprc_per_label(y, p)

    rep = rates(y, states == PRESENT)
    t["report_threshold"] = ops.report
    for k in ("sens", "spec", "ppv", "npv"):
        t[f"report_{k}"] = rep[k]
    rule = rates(y, states != ABSENT)  # "not absent" = not ruled out
    t["ruleout_threshold"] = ops.ruleout
    t["ruleout_sens"], t["ruleout_npv"] = rule["sens"], rule["npv"]
    t["uncertain_rate"] = _div(((states == UNCERTAIN) & known).sum(0), known.sum(0))
    mf = rates(y, p >= ops.max_f1)
    t["maxf1_threshold"] = ops.max_f1
    for k in ("sens", "spec", "ppv"):
        t[f"maxf1_{k}"] = mf[k]
    for s, thr in ops.at_specificity.items():
        r = rates(y, p >= thr)
        t[f"sens_at_spec{round(s * 100)}"], t[f"spec_at_spec{round(s * 100)}"] = r["sens"], r["spec"]
    t["ece_raw"] = ece_per_label(y, p_raw, cfg.calibration_bins)
    t["ece"] = ece_per_label(y, p, cfg.calibration_bins)
    t["brier"] = brier_per_label(y, p)
    for pt in cfg.net_benefit_thresholds:
        nb, nb_all = net_benefit(y, p, pt)
        t[f"net_benefit@{pt:g}"], t[f"net_benefit_all@{pt:g}"] = nb, nb_all
    return t


# ------------------------------------------------------------------ per scan
_RULES = ("report", "maxf1")


def per_scan(ids: pd.DataFrame, y: np.ndarray, p: np.ndarray, ops: OperatingPoints) -> pd.DataFrame:
    """One row per volume: what a reader would get for that scan under each rule."""
    known = ~np.isnan(y)
    pos = known & (y == 1)
    neg = known & (y == 0)
    states = ops.states(p)
    out = ids.reset_index(drop=True).copy()
    out["n_known"] = known.sum(1)
    out["n_true"] = pos.sum(1)
    for rule, pred in (("report", states == PRESENT), ("maxf1", p >= ops.max_f1)):
        pred = pred & known
        out[f"{rule}_pred"] = pred.sum(1)
        out[f"{rule}_tp"] = (pred & pos).sum(1)
        out[f"{rule}_fp"] = (pred & neg).sum(1)
        out[f"{rule}_fn"] = (~pred & pos).sum(1)
        out[f"{rule}_exact"] = ((pred == pos) | ~known).all(1)
    out["n_uncertain"] = ((states == UNCERTAIN) & known).sum(1)
    return out


def scan_summary(scans: pd.DataFrame) -> dict:
    normal = scans.n_true == 0
    row = {"n_volumes": int(len(scans)), "n_normal": int(normal.sum()), "true_per_scan": float(scans.n_true.mean())}
    for rule in _RULES:
        row[f"{rule}_pred_per_scan"] = float(scans[f"{rule}_pred"].mean())
        row[f"{rule}_tp_per_scan"] = float(scans[f"{rule}_tp"].mean())
        row[f"{rule}_fp_per_scan"] = float(scans[f"{rule}_fp"].mean())
        row[f"{rule}_fn_per_scan"] = float(scans[f"{rule}_fn"].mean())
        row[f"{rule}_scans_with_fp"] = float((scans[f"{rule}_fp"] > 0).mean())
        row[f"{rule}_exact_match"] = float(scans[f"{rule}_exact"].mean())
        row[f"{rule}_normal_flag_rate"] = float((scans.loc[normal, f"{rule}_pred"] > 0).mean()) if normal.any() else np.nan
    row["uncertain_per_scan"] = float(scans.n_uncertain.mean())
    row["scans_with_uncertain"] = float((scans.n_uncertain > 0).mean())
    return row


# ------------------------------------------------------------------ paired reconstructions
def kernel_pairs(ids: pd.DataFrame) -> pd.DataFrame:
    """Row positions of (sharp, soft) reconstructions of the same scan: columns scan, sharp, soft.
    Needs ``patient_id``, ``scan_id`` and ``kernel_class``; empty if any is missing."""
    if not {"patient_id", "scan_id", "kernel_class"} <= set(ids.columns):
        return pd.DataFrame(columns=["scan", "sharp", "soft"])
    frame = ids.reset_index(drop=True)
    frame = frame.assign(_row=np.arange(len(frame)), scan=frame.patient_id.astype(str) + "/" + frame.scan_id.astype(str))
    frame = frame[frame.kernel_class.isin(PAIR_KERNELS) & frame.scan_id.notna()]
    first = frame.drop_duplicates(["scan", "kernel_class"]).pivot(index="scan", columns="kernel_class", values="_row")
    if not set(PAIR_KERNELS) <= set(first.columns):
        return pd.DataFrame(columns=["scan", "sharp", "soft"])
    return first[list(PAIR_KERNELS)].dropna().astype(int).reset_index()


def paired(ids: pd.DataFrame, p: np.ndarray, ops: OperatingPoints) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(per-label table, per-pair table, summary) of decision changes between the two kernels."""
    pairs = kernel_pairs(ids)
    names = ops.label_names
    if pairs.empty:
        empty = pd.DataFrame({"label": names, "n_pairs": 0})
        return empty, pd.DataFrame(), {"n_pairs": 0}
    a, b = pairs.sharp.to_numpy(), pairs.soft.to_numpy()
    states = ops.states(p)
    rep = states == PRESENT
    mf = p >= ops.max_f1
    flips, mf_flips, changes = rep[a] != rep[b], mf[a] != mf[b], states[a] != states[b]
    # a hard contradiction: "present" on one kernel and "absent" on the other (not just a move into "uncertain")
    hard = ((states[a] == PRESENT) & (states[b] == ABSENT)) | ((states[a] == ABSENT) & (states[b] == PRESENT))
    dp = p[b] - p[a]
    table = pd.DataFrame({"label": names, "n_pairs": len(pairs), "flip_rate": flips.mean(0),
                          "maxf1_flip_rate": mf_flips.mean(0), "state_change_rate": changes.mean(0),
                          "hard_contradiction_rate": hard.mean(0),
                          "mean_abs_dp": np.abs(dp).mean(0), "mean_dp_soft_minus_sharp": dp.mean(0)})
    per_pair = pairs.assign(sharp_volume=ids.volume_id.to_numpy()[a], soft_volume=ids.volume_id.to_numpy()[b],
                            flips=flips.sum(1), maxf1_flips=mf_flips.sum(1), state_changes=changes.sum(1),
                            hard_contradictions=hard.sum(1))
    summary = {"n_pairs": int(len(pairs)),
               "paired_scans_with_flip": float((flips.sum(1) > 0).mean()),
               "paired_flips_per_scan": float(flips.sum(1).mean()),
               "paired_maxf1_scans_with_flip": float((mf_flips.sum(1) > 0).mean()),
               "paired_maxf1_flips_per_scan": float(mf_flips.sum(1).mean()),
               "paired_state_changes_per_scan": float(changes.sum(1).mean()),
               "paired_scans_with_hard_contradiction": float((hard.sum(1) > 0).mean()),
               "paired_hard_contradictions_per_scan": float(hard.sum(1).mean())}
    return table, per_pair.drop(columns=["sharp", "soft"]), summary


# ------------------------------------------------------------------ gate
def gate(labels_all: pd.DataFrame, kernel_gap: pd.Series | None, paired_table: pd.DataFrame,
         cfg: ClinicalConfig) -> pd.DataFrame:
    """Per label: each rule's value and pass / fail (NaN = n/a), and the overall status."""
    g = cfg.gate
    t = labels_all[["label"]].copy()
    t["auroc_ci_low"] = labels_all.auroc_ci_low.to_numpy()
    t["report_sens"] = labels_all.report_sens.to_numpy()
    t["ece"] = labels_all.ece.to_numpy()
    flip = paired_table.set_index("label")["flip_rate"] if "flip_rate" in paired_table else pd.Series(dtype=float)
    t["paired_flip_rate"] = t.label.map(flip).astype(float)
    t["kernel_sens_gap"] = t.label.map(kernel_gap).astype(float) if kernel_gap is not None else np.nan

    def check(values, ok):
        values = np.asarray(values, dtype=float)
        return np.where(np.isnan(values), np.nan, ok(values).astype(float))

    t["pass_auroc"] = check(t.auroc_ci_low, lambda v: v >= g.auroc_ci_low)
    t["pass_sens"] = check(t.report_sens, lambda v: v >= g.report_sensitivity)
    t["pass_ece"] = check(t.ece, lambda v: v <= g.ece)
    t["pass_flip"] = check(t.paired_flip_rate, lambda v: v <= g.paired_flip_rate)
    t["pass_kernel_gap"] = check(t.kernel_sens_gap, lambda v: np.abs(v) <= g.kernel_sensitivity_gap)
    discrimination = t[["pass_auroc", "pass_sens"]].eq(0).any(axis=1)
    robustness = t[["pass_ece", "pass_flip", "pass_kernel_gap"]].eq(0).any(axis=1)
    t["status"] = np.where(discrimination, NOT_SUPPORTED, np.where(robustness, REVIEW, SUPPORTED))
    return t
