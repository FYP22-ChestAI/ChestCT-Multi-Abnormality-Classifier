"""Operating points and calibration, chosen on VAL only and applied unchanged to the scored split.

* Calibration (per label, on logit(p)): ``platt`` p' = sigmoid(a * logit(p) + b), ``temperature``
  (b = 0), or ``none``. Optionally a separate fit per kernel class, where VAL has enough volumes of it;
  other kernels use the global fit. A fit with a non-positive slope is replaced by the identity, so
  calibration never reverses a label's ranking.
* Thresholds, per label, on the calibrated VAL probabilities (a volume is positive when p >= t):
    report     the lowest t whose VAL specificity >= ``report_specificity``     -> "present"
    ruleout    the highest t whose VAL sensitivity >= ``ruleout_sensitivity``   -> below it "absent"
    max_f1     the legacy phase-1 threshold (ct_model.training.metrics.max_f1_thresholds)
    spec@s     the lowest t whose VAL specificity >= s, for each ``fixed_specificities`` s
  If ruleout > report the uncertain band is empty (ruleout is set to report).
* Three outcomes per (volume, label): ABSENT (p < ruleout), UNCERTAIN, PRESENT (p >= report).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ..training import metrics as M
from .config import ClinicalConfig

ABSENT, UNCERTAIN, PRESENT = 0, 1, 2
_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _fit_one(z: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    """(slope, intercept) minimising the log loss of sigmoid(slope * z + intercept) against y."""
    if method == "none" or y.size == 0 or y.min() == y.max():
        return 1.0, 0.0

    def nll(params):
        a, b = (params[0], 0.0) if method == "temperature" else params
        s = a * z + b
        return float(np.mean(np.logaddexp(0.0, s) - y * s))

    x0 = [1.0] if method == "temperature" else [1.0, 0.0]
    res = minimize(nll, x0, method="L-BFGS-B")
    a, b = (float(res.x[0]), 0.0) if method == "temperature" else (float(res.x[0]), float(res.x[1]))
    if not res.success or not math.isfinite(a) or not math.isfinite(b) or a <= 0:
        return 1.0, 0.0
    return a, b


def _fit_params(y: np.ndarray, p: np.ndarray, method: str) -> np.ndarray:
    z = logit(p)
    out = np.empty((y.shape[1], 2))
    for c in range(y.shape[1]):
        keep = ~np.isnan(y[:, c])
        out[c] = _fit_one(z[keep, c], y[keep, c], method)
    return out


@dataclass
class Calibration:
    method: str
    params: np.ndarray                                   # (C, 2): slope, intercept per label
    per_kernel: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def fit(cls, y: np.ndarray, p: np.ndarray, kernels: np.ndarray | None, method: str,
            by_kernel: bool = False, min_volumes: int = 100) -> "Calibration":
        cal = cls(method, _fit_params(y, p, method))
        if by_kernel and kernels is not None and method != "none":
            for kc in sorted(set(kernels)):
                rows = kernels == kc
                if rows.sum() >= min_volumes:
                    cal.per_kernel[str(kc)] = _fit_params(y[rows], p[rows], method)
        return cal

    def apply(self, p: np.ndarray, kernels: np.ndarray | None = None) -> np.ndarray:
        z = logit(p)
        out = sigmoid(z * self.params[:, 0] + self.params[:, 1])
        if self.per_kernel and kernels is not None:
            for kc, params in self.per_kernel.items():
                rows = kernels == kc
                out[rows] = sigmoid(z[rows] * params[:, 0] + params[:, 1])
        return out

    def to_dict(self, names: list[str]) -> dict:
        def table(params):
            return {n: {"slope": float(a), "intercept": float(b)} for n, (a, b) in zip(names, params)}
        return {"method": self.method, "global": table(self.params),
                "per_kernel": {kc: table(p) for kc, p in self.per_kernel.items()}}


def threshold_at_specificity(y: np.ndarray, p: np.ndarray, specificity: float) -> float:
    """Lowest t with mean(p[negatives] < t) >= specificity (positive = p >= t); NaN if not scorable."""
    neg = np.sort(p[y == 0])
    if neg.size == 0 or (y == 1).sum() == 0:
        return float("nan")
    k = math.ceil(specificity * neg.size - 1e-9)
    return float(np.nextafter(neg[k - 1], np.inf)) if k > 0 else float(neg[0])


def threshold_at_sensitivity(y: np.ndarray, p: np.ndarray, sensitivity: float) -> float:
    """Highest t with mean(p[positives] >= t) >= sensitivity; NaN if not scorable."""
    pos = np.sort(p[y == 1])
    if pos.size == 0 or (y == 0).sum() == 0:
        return float("nan")
    m = max(1, math.ceil(sensitivity * pos.size - 1e-9))
    return float(pos[pos.size - m])


def _per_label(y: np.ndarray, p: np.ndarray, fn, target: float) -> np.ndarray:
    out = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        t, q = M._known(y, p, c)
        out[c] = fn(t, q, target)
    return out


@dataclass
class OperatingPoints:
    """Everything chosen on VAL: the calibration and every threshold, per label."""
    label_names: list[str]
    calibration: Calibration
    report: np.ndarray
    ruleout: np.ndarray
    max_f1: np.ndarray
    at_specificity: dict[float, np.ndarray]

    def calibrate(self, p: np.ndarray, kernels: np.ndarray | None = None) -> np.ndarray:
        return self.calibration.apply(p, kernels)

    def states(self, p_cal: np.ndarray) -> np.ndarray:
        """(N, C) of ABSENT / UNCERTAIN / PRESENT, from CALIBRATED probabilities."""
        s = np.full(p_cal.shape, UNCERTAIN, dtype=np.int8)
        s[p_cal >= self.report] = PRESENT
        s[p_cal < self.ruleout] = ABSENT
        return s

    def to_dict(self, specificity: float, sensitivity: float) -> dict:
        def per_label(values):
            return {n: (None if np.isnan(v) else float(v)) for n, v in zip(self.label_names, values)}
        return {
            "chosen_on": "val",
            "report_specificity": specificity, "ruleout_sensitivity": sensitivity,
            "calibration": self.calibration.to_dict(self.label_names),
            "thresholds": {"report": per_label(self.report), "ruleout": per_label(self.ruleout),
                           "max_f1": per_label(self.max_f1),
                           **{f"spec{round(s * 100)}": per_label(t) for s, t in self.at_specificity.items()}},
        }


def choose_operating_points(y_val: np.ndarray, p_val: np.ndarray, label_names: list[str], cfg: ClinicalConfig,
                            kernels_val: np.ndarray | None = None) -> OperatingPoints:
    op = cfg.operating_points
    cal = Calibration.fit(y_val, p_val, kernels_val, op.calibration, op.calibration_by_kernel, op.min_calibration_volumes)
    q = cal.apply(p_val, kernels_val)
    report = _per_label(y_val, q, threshold_at_specificity, op.report_specificity)
    ruleout = _per_label(y_val, q, threshold_at_sensitivity, op.ruleout_sensitivity)
    ruleout = np.where(ruleout > report, report, ruleout)
    return OperatingPoints(
        label_names=list(label_names), calibration=cal, report=report, ruleout=ruleout,
        max_f1=M.max_f1_thresholds(y_val, q),
        at_specificity={s: _per_label(y_val, q, threshold_at_specificity, s) for s in cfg.fixed_specificities},
    )
