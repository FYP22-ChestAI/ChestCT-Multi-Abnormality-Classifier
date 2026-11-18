"""The clinical evaluation: operating points from val, per-scan counts, paired kernels, gate, reports."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

from ct_model.clinical import scorecard as SC
from ct_model.clinical.config import ClinicalConfig, parse_clinical_config
from ct_model.clinical.operating import (
    ABSENT, PRESENT, UNCERTAIN, Calibration, choose_operating_points, threshold_at_sensitivity,
    threshold_at_specificity,
)
from ct_model.clinical.report import clinical_report
from ct_model.training.results import load_results

NAMES = ["nodule", "effusion"]


def _cfg(**kw) -> ClinicalConfig:
    cfg = ClinicalConfig(bootstrap=0, min_group=1)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def _scores(n, seed, shift=1.5):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 2)) < 0.3).astype(float)
    p = 1 / (1 + np.exp(-(rng.normal(size=(n, 2)) + shift * y - 1)))
    return y, p


def test_thresholds_reach_their_targets_on_val():
    y, p = _scores(500, 0)
    for c in range(2):
        t = threshold_at_specificity(y[:, c], p[:, c], 0.95)
        assert np.mean(p[y[:, c] == 0, c] < t) >= 0.95
        assert np.mean(p[y[:, c] == 0, c] < np.nextafter(t, -np.inf)) < 0.95 + 1e-9  # the lowest such threshold
        t = threshold_at_sensitivity(y[:, c], p[:, c], 0.90)
        assert np.mean(p[y[:, c] == 1, c] >= t) >= 0.90
    assert np.isnan(threshold_at_specificity(np.zeros(5), np.linspace(0, 1, 5), 0.9))  # no positives: not scorable


def test_three_states_and_ruleout_never_above_report():
    y, p = _scores(400, 1)
    ops = choose_operating_points(y, p, NAMES, _cfg())
    assert (ops.ruleout <= ops.report).all()
    q = ops.calibrate(p)
    s = ops.states(q)
    assert set(np.unique(s)) <= {ABSENT, UNCERTAIN, PRESENT}
    assert ((s == PRESENT) == (q >= ops.report)).all()
    assert ((s == ABSENT) == (q < ops.ruleout)).all()


def test_calibration_keeps_the_ranking_and_max_f1_decisions():
    y, p = _scores(600, 2, shift=2.5)
    raw = choose_operating_points(y, p, NAMES, _cfg(operating_points=parse_clinical_config({"operating_points": {"calibration": "none"}}).operating_points))
    cal = choose_operating_points(y, p, NAMES, _cfg())
    y_t, p_t = _scores(300, 3, shift=2.5)
    assert ((p_t >= raw.max_f1) == (cal.calibrate(p_t) >= cal.max_f1)).all()
    assert np.allclose(SC.auroc_columns(y_t, p_t), SC.auroc_columns(y_t, cal.calibrate(p_t)))
    # per-kernel calibration only where val has enough volumes of that kernel
    kernels = np.array(["sharp"] * 500 + ["soft"] * 100)
    c = Calibration.fit(y, p, kernels, "platt", by_kernel=True, min_volumes=200)
    assert list(c.per_kernel) == ["sharp"]


def test_auroc_columns_matches_sklearn():
    y, p = _scores(200, 4)
    assert np.allclose(SC.auroc_columns(y, p), [roc_auc_score(y[:, c], p[:, c]) for c in range(2)])


def test_per_scan_counts_false_positives_and_missed_findings():
    y, p = _scores(400, 5)
    ops = choose_operating_points(y, p, NAMES, _cfg())
    ops.report = np.array([0.5, 0.5])
    ops.ruleout = np.array([0.2, 0.2])
    ops.max_f1 = np.array([0.3, 0.3])
    y = np.array([[1, 0], [0, 0], [1, np.nan]])
    q = np.array([[0.9, 0.6], [0.1, 0.4], [0.3, 0.9]])
    ids = pd.DataFrame({"volume_id": ["a", "b", "c"]})
    scans = SC.per_scan(ids, y, q, ops)
    assert scans.report_tp.tolist() == [1, 0, 0]
    assert scans.report_fp.tolist() == [1, 0, 0]
    assert scans.report_fn.tolist() == [0, 0, 1]
    assert scans.report_pred.tolist() == [2, 0, 0]  # the unknown label of c is never counted
    assert scans.report_exact.tolist() == [False, True, False]
    assert scans.maxf1_fp.tolist() == [1, 1, 0]
    assert scans.n_uncertain.tolist() == [0, 1, 1]
    s = SC.scan_summary(scans)
    assert s["n_normal"] == 1 and s["report_normal_flag_rate"] == 0.0 and s["maxf1_normal_flag_rate"] == 1.0
    assert s["report_fp_per_scan"] == pytest.approx(1 / 3)


def test_paired_kernels_count_decision_changes():
    y, p = _scores(200, 6)
    ops = choose_operating_points(y, p, NAMES, _cfg())
    ops.report = np.array([0.5, 0.5])
    ops.ruleout = np.array([0.5, 0.5])
    ops.max_f1 = np.array([0.5, 0.5])
    ids = pd.DataFrame({"volume_id": ["p1_a_1", "p1_a_2", "p2_a_1", "p2_a_2", "p3_a_1"],
                        "patient_id": ["p1", "p1", "p2", "p2", "p3"], "scan_id": ["a"] * 5,
                        "kernel_class": ["sharp", "soft", "soft", "sharp", "sharp"]})
    q = np.array([[0.9, 0.9], [0.1, 0.9], [0.6, 0.2], [0.7, 0.2], [0.9, 0.9]])
    table, pairs, summary = SC.paired(ids, q, ops)
    assert summary["n_pairs"] == 2  # p3 has no soft reconstruction
    assert table.flip_rate.tolist() == [0.5, 0.0]
    assert summary["paired_scans_with_flip"] == 0.5 and summary["paired_flips_per_scan"] == 0.5
    # p1 nodule: present (0.9) vs absent (0.1) is a hard contradiction; p2 nodule 0.7 vs 0.6 stays present on both
    assert table.hard_contradiction_rate.tolist() == [0.5, 0.0] and summary["paired_scans_with_hard_contradiction"] == 0.5
    assert np.isclose(table.mean_dp_soft_minus_sharp[0], ((0.1 - 0.9) + (0.6 - 0.7)) / 2)
    assert SC.paired(ids.drop(columns="scan_id"), q, ops)[2] == {"n_pairs": 0}


def test_gate_separates_discrimination_from_robustness():
    labels = pd.DataFrame({"label": ["a", "b", "c", "d"], "auroc_ci_low": [0.9, 0.9, 0.7, 0.9],
                           "report_sens": [0.8, 0.8, 0.8, np.nan], "ece": [0.01, 0.2, 0.01, 0.01]})
    paired_table = pd.DataFrame({"label": ["a", "b", "c", "d"], "flip_rate": [0.05, 0.05, 0.05, 0.05]})
    gap = pd.Series({"a": -0.05, "b": 0.0, "c": 0.0, "d": 0.0})
    g = SC.gate(labels, gap, paired_table, ClinicalConfig())
    assert g.status.tolist() == [SC.SUPPORTED, SC.REVIEW, SC.NOT_SUPPORTED, SC.SUPPORTED]  # NaN = n/a, not a fail
    assert np.isnan(g.pass_sens[3])


def test_config_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValueError, match="unknown key"):
        parse_clinical_config({"gate": {"auroc": 0.8}})
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_clinical_config({"operating_points": {"report_specificity": 95}})
    assert parse_clinical_config({}).fingerprint() == ClinicalConfig().fingerprint()


# ------------------------------------------------------------------ reports from run folders
def _frame(y, p, ids):
    frame = ids.copy()
    for c, n in enumerate(NAMES):
        frame[f"prob_{n}"] = p[:, c]
        frame[f"true_{n}"] = y[:, c]
    return frame


def _fake_run(root, seed, y_val, y_test, noise):
    folder = root / "exp" / "aaaa1111" / f"seed{seed}"
    (folder / "eval" / "ctrate__r1__test").mkdir(parents=True)
    rng = np.random.default_rng(seed)
    p_val = np.clip(0.6 * y_val + rng.uniform(0, noise, y_val.shape), 0, 1)
    p_test = np.clip(0.6 * y_test + rng.uniform(0, noise, y_test.shape), 0, 1)
    n_val, n_test = len(y_val), len(y_test)
    val_ids = pd.DataFrame({"volume_id": [f"v{i}" for i in range(n_val)], "patient_id": [f"v{i}" for i in range(n_val)],
                            "split": "val", "kernel_class": "sharp"})
    pats = [f"t{i // 2}" for i in range(n_test)]
    test_ids = pd.DataFrame({"volume_id": [f"{p}_a_{i % 2 + 1}" for i, p in enumerate(pats)], "patient_id": pats,
                             "split": "test", "kernel_class": ["sharp", "soft"] * (n_test // 2), "scan_id": "a",
                             "manufacturer": "X"})
    _frame(y_val, p_val, val_ids).to_csv(folder / "predictions_val.csv", index=False)
    _frame(y_test, p_test, test_ids).to_csv(folder / "eval" / "ctrate__r1__test" / "predictions.csv", index=False)
    (folder / "labels.json").write_text(json.dumps(NAMES))
    (folder / "config.yaml").write_text(yaml.safe_dump({"name": "exp", "seed": seed, "output_dir": root.as_posix(),
                                                        "data_config": (root / "missing.yaml").as_posix(),
                                                        "data": {"source": "ctrate", "run": "r1"}}))
    (folder / "run_info.json").write_text(json.dumps({"run_id": f"exp/aaaa1111/seed{seed}", "config_hash": "aaaa1111bbbb2222",
                                                      "status": "finished"}))
    return folder


def test_reports_for_a_run_and_a_seed_ensemble(tmp_path):
    rng = np.random.default_rng(0)
    y_val = (rng.random((300, 2)) < 0.3).astype(float)
    y_test = np.repeat((rng.random((100, 2)) < 0.3).astype(float), 2, axis=0)  # both kernels share the labels
    runs = [_fake_run(tmp_path, s, y_val, y_test, noise=0.7) for s in (0, 1)]
    cfg = _cfg(min_group=10)
    logs = []

    [one] = clinical_report(runs[0], config=cfg, log=logs.append)
    assert one.dir == runs[0] / "clinical" / "ctrate__r1__test"
    for f in ("scorecard.json", "per_label.csv", "summary.csv", "per_scan.csv", "paired_kernel.csv", "gate.csv",
              "operating_points.json", "predictions_clinical.csv"):
        assert (one.dir / f).is_file(), f
    h = one.scorecard["headline"]
    assert h["n_volumes"] == 200 and h["n_pairs"] == 100 and h["n_normal"] == int((y_test.sum(1) == 0).sum())
    assert set(one.per_label.group_by) == {"all", "kernel_class", "manufacturer"}
    assert any("could not read the run manifest" in m for m in logs)  # no manifest: still reported from the CSVs

    results = clinical_report(runs[0].parent, config=cfg, per_seed=True, log=logs.append)
    assert len(results) == 3 and results[-1].dir == runs[0].parent / "ensemble" / "clinical" / "ctrate__r1__test"
    card = results[-1].scorecard
    assert card["seed"] == "ensemble" and card["members"] == 2 and card["run_id"] == "exp/aaaa1111/ensemble"
    ens = pd.read_csv(results[-1].dir / "predictions.csv").set_index("volume_id")
    a = pd.read_csv(runs[0] / "eval" / "ctrate__r1__test" / "predictions.csv").set_index("volume_id")
    b = pd.read_csv(runs[1] / "eval" / "ctrate__r1__test" / "predictions.csv").set_index("volume_id")
    assert np.allclose(ens.prob_nodule, (a.prob_nodule + b.loc[a.index].prob_nodule) / 2)

    ledger = pd.read_csv(tmp_path / "clinical.csv")
    assert len(ledger) == 4 and ledger.run_id.iloc[-1] == "exp/aaaa1111/ensemble"
    assert {"report_fp_per_scan", "paired_scans_with_flip", "labels_supported"} <= set(ledger.columns)


def test_results_loader_reads_clinical_reports(tmp_path):
    rng = np.random.default_rng(1)
    y_val = (rng.random((200, 2)) < 0.3).astype(float)
    y_test = np.repeat((rng.random((60, 2)) < 0.3).astype(float), 2, axis=0)
    run = _fake_run(tmp_path, 0, y_val, y_test, noise=0.7)
    (run / "val_metrics.json").write_text(json.dumps({"macro_auroc": 0.8}))
    clinical_report(run.parent, config=_cfg(min_group=10), per_seed=True, log=lambda m: None)
    res = load_results(tmp_path)
    assert len(res.clinical) == 2 and set(res.clinical.ensemble) == {True, False}
    assert {"finding", "status", "group_by", "report_sens"} <= set(res.clinical_labels.columns)
