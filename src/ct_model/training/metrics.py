"""Evaluation metrics for multi-label prediction. Every score comes from scikit-learn; this module
only decides what is computed on which data.

* Threshold-free, per label: AUROC (``roc_auc_score``) and AUPRC (``average_precision_score``).
  Summaries: macro = mean over labels that can be scored, micro = AUROC over all (volume, label) pairs.
* Thresholded: one threshold per label, chosen on VAL as the one maximising F1 (from
  ``precision_recall_curve``), then applied unchanged to any other split -- never tuned on test.
  Gives F1, sensitivity (recall), specificity, PPV (precision).
* Uncertainty: 95 % bootstrap CI of a summary metric, resampling PATIENTS (all volumes of a drawn
  patient together, since two reconstructions of one scan are not independent), seeded.

A label with only one class present in the data being scored gets NaN -- never silently dropped
from a table, and left out of the macro mean (the count of scored labels is reported with it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _scorable(y: np.ndarray) -> bool:
    y = y[~np.isnan(y)]
    return y.size > 0 and 0 < y.sum() < y.size


def _known(y_true: np.ndarray, y_prob: np.ndarray, c: int) -> tuple[np.ndarray, np.ndarray]:
    keep = ~np.isnan(y_true[:, c])
    return y_true[keep, c], y_prob[keep, c]


def auroc_per_label(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    out = np.full(y_true.shape[1], np.nan)
    for c in range(y_true.shape[1]):
        t, p = _known(y_true, y_prob, c)
        if _scorable(t):
            out[c] = roc_auc_score(t, p)
    return out


def auprc_per_label(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    out = np.full(y_true.shape[1], np.nan)
    for c in range(y_true.shape[1]):
        t, p = _known(y_true, y_prob, c)
        if _scorable(t):
            out[c] = average_precision_score(t, p)
    return out


def micro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    keep = ~np.isnan(y_true)
    t, p = y_true[keep], y_prob[keep]
    return float(roc_auc_score(t, p)) if _scorable(t) else float("nan")


def max_f1_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per label, the probability threshold with the highest F1 on these data (meant for VAL)."""
    out = np.full(y_true.shape[1], np.nan)
    for c in range(y_true.shape[1]):
        t, p = _known(y_true, y_prob, c)
        if not _scorable(t):
            continue
        precision, recall, thresholds = precision_recall_curve(t, p)
        precision, recall = precision[:-1], recall[:-1]  # the last point has no threshold
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
        out[c] = thresholds[int(np.argmax(f1))]
    return out


def thresholded(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Per label: F1, sensitivity, specificity, PPV at the given thresholds (predict positive if p >= t)."""
    rows = []
    for c in range(y_true.shape[1]):
        t, p = _known(y_true, y_prob, c)
        if np.isnan(thresholds[c]) or t.size == 0:
            rows.append({"f1": np.nan, "sensitivity": np.nan, "specificity": np.nan, "ppv": np.nan})
            continue
        pred = p >= thresholds[c]
        tp, fp = int((pred & (t == 1)).sum()), int((pred & (t == 0)).sum())
        fn, tn = int((~pred & (t == 1)).sum()), int((~pred & (t == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
        rows.append({"f1": f1, "sensitivity": sens, "specificity": spec, "ppv": ppv})
    return pd.DataFrame(rows)


def per_label_table(y_true: np.ndarray, y_prob: np.ndarray, label_names: list[str],
                    thresholds: np.ndarray | None = None) -> pd.DataFrame:
    known = ~np.isnan(y_true)
    table = pd.DataFrame({
        "label": label_names,
        "n": known.sum(0),
        "n_pos": np.nansum(y_true, axis=0).astype(int),
        "auroc": auroc_per_label(y_true, y_prob),
        "auprc": auprc_per_label(y_true, y_prob),
    })
    if thresholds is not None:
        table["threshold"] = thresholds
        table = pd.concat([table, thresholded(y_true, y_prob, thresholds)], axis=1)
    return table


def summary(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray | None = None) -> dict:
    au, ap = auroc_per_label(y_true, y_prob), auprc_per_label(y_true, y_prob)
    out = {
        "n_volumes": int(y_true.shape[0]),
        "labels_scored": int((~np.isnan(au)).sum()),
        "macro_auroc": float(np.nanmean(au)) if (~np.isnan(au)).any() else float("nan"),
        "macro_auprc": float(np.nanmean(ap)) if (~np.isnan(ap)).any() else float("nan"),
        "micro_auroc": micro_auroc(y_true, y_prob),
    }
    if thresholds is not None:
        t = thresholded(y_true, y_prob, thresholds)
        out.update({f"macro_{k}": float(t[k].mean()) for k in ("f1", "sensitivity", "specificity", "ppv")})
    return out


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    au = auroc_per_label(y_true, y_prob)
    return float(np.nanmean(au)) if (~np.isnan(au)).any() else float("nan")


def macro_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    ap = auprc_per_label(y_true, y_prob)
    return float(np.nanmean(ap)) if (~np.isnan(ap)).any() else float("nan")


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, patient_ids, metric=macro_auroc,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of ``metric`` over ``n_boot`` resamples of PATIENTS (with replacement)."""
    patient_ids = np.asarray(patient_ids)
    patients, inverse = np.unique(patient_ids, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == k) for k in range(len(patients))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        drawn = rng.integers(0, len(patients), size=len(patients))
        idx = np.concatenate([rows_of[k] for k in drawn])
        values.append(metric(y_true[idx], y_prob[idx]))
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))
