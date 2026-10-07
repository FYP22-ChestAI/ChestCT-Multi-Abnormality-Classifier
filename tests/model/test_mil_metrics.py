"""Metrics against scikit-learn and brute force: AUROC / AUPRC, max-F1 thresholds, thresholded counts,
and the patient-level bootstrap."""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ct_model.training import metrics as M

RNG = np.random.default_rng(0)
Y = RNG.integers(0, 2, size=(200, 3)).astype(float)
P = np.clip(Y * 0.3 + RNG.random((200, 3)) * 0.7, 0, 1)


def test_per_label_scores_equal_sklearn():
    for c in range(3):
        assert M.auroc_per_label(Y, P)[c] == pytest.approx(roc_auc_score(Y[:, c], P[:, c]))
        assert M.auprc_per_label(Y, P)[c] == pytest.approx(average_precision_score(Y[:, c], P[:, c]))
    assert M.micro_auroc(Y, P) == pytest.approx(roc_auc_score(Y.ravel(), P.ravel()))
    s = M.summary(Y, P)
    assert s["macro_auroc"] == pytest.approx(np.mean([roc_auc_score(Y[:, c], P[:, c]) for c in range(3)]))
    assert M.macro_auprc(Y, P) == pytest.approx(np.mean([average_precision_score(Y[:, c], P[:, c]) for c in range(3)]))
    assert s["macro_auprc"] == pytest.approx(M.macro_auprc(Y, P))
    assert np.isnan(M.macro_auprc(np.zeros((4, 2)), np.random.rand(4, 2)))  # nothing scorable: NaN, no warning


def test_single_class_label_is_nan_not_dropped():
    y = Y.copy()
    y[:, 1] = 0
    au = M.auroc_per_label(y, P)
    assert np.isnan(au[1]) and not np.isnan(au[[0, 2]]).any()
    s = M.summary(y, P)
    assert s["labels_scored"] == 2 and s["macro_auroc"] == pytest.approx(np.nanmean(au))
    table = M.per_label_table(y, P, ["a", "b", "c"])
    assert len(table) == 3 and np.isnan(table.auroc[1])


def test_missing_labels_are_ignored():
    y = Y.copy()
    y[:50, 0] = np.nan
    assert M.auroc_per_label(y, P)[0] == pytest.approx(roc_auc_score(Y[50:, 0], P[50:, 0]))


def test_thresholds_maximise_f1_by_brute_force():
    t = M.max_f1_thresholds(Y, P)
    for c in range(3):
        best = max(f1_score(Y[:, c], P[:, c] >= cand) for cand in np.unique(P[:, c]))
        assert f1_score(Y[:, c], P[:, c] >= t[c]) == pytest.approx(best)


def test_thresholded_counts_by_hand():
    y = np.array([[1], [1], [0], [0], [0]], dtype=float)
    p = np.array([[0.9], [0.4], [0.6], [0.2], [0.1]])
    row = M.thresholded(y, p, np.array([0.5])).iloc[0]
    # predicted positive: 0.9 (TP), 0.6 (FP); negatives: 0.4 (FN), 0.2, 0.1 (TN)
    assert row.sensitivity == 0.5 and row.specificity == pytest.approx(2 / 3) and row.ppv == 0.5
    assert row.f1 == pytest.approx(f1_score(y[:, 0], p[:, 0] >= 0.5))


def test_bootstrap_is_seeded_brackets_the_estimate_and_resamples_patients():
    patients = np.repeat(np.arange(100), 2)  # two volumes per patient
    lo, hi = M.bootstrap_ci(Y, P, patients, n_boot=200, seed=3)
    assert (lo, hi) == M.bootstrap_ci(Y, P, patients, n_boot=200, seed=3)
    assert lo <= M.macro_auroc(Y, P) <= hi


def test_bootstrap_draws_whole_patients():
    # patient 0 owns rows 0-9; patients 1..20 own one row each. Row ids ride along in the probabilities.
    patients = np.array([0] * 10 + list(range(1, 21)))
    rows = np.arange(30, dtype=float)[:, None]
    draws = []
    M.bootstrap_ci(np.zeros((30, 1)), rows, patients, metric=lambda y, p: draws.append(p[:, 0].astype(int)) or 0.0,
                   n_boot=50, seed=1)
    for d in draws:
        n_from_p0 = np.isin(d, np.arange(10)).sum()
        assert n_from_p0 % 10 == 0                       # patient 0's volumes always come all together
        for k in range(10):                              # ... the same number of times each
            assert (d == k).sum() == n_from_p0 // 10
    assert len({len(d) for d in draws}) > 1              # resampling patients, not volumes: sizes vary
