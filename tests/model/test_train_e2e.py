"""Stages 3 + 4 end to end, through the real scripts, on a synthetic embedding store with PLANTED findings:
a volume has label c iff a few consecutive slices carry an extra signal along direction e_c. That lets
the tests check that the model learns, that test is never touched by training, that results are
recorded identifiably, that a run resumes, and that the slice evidence points at the planted slices."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "model"
LABELS = ["nodule", "effusion", "emphysema"]  # the mil_project fixture (conftest.py)


def call(monkeypatch, name, *args):
    spec = importlib.util.spec_from_file_location(f"model_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", [f"{name}.py", *map(str, args)])
    return mod.main()


def _only_run_dir(outputs: Path) -> Path:
    (d,) = [p for p in outputs.glob("synth_abmil/*/seed*") if p.is_dir()]
    return d


def test_train_evaluate_summarize_and_evidence(mil_project, monkeypatch, capsys):
    from ct_model.training import data as data_mod

    requested = []
    original = data_mod.ExperimentData.dataset
    monkeypatch.setattr(data_mod.ExperimentData, "dataset",
                        lambda self, split, *a, **k: requested.append(split) or original(self, split, *a, **k))

    assert call(monkeypatch, "train_mil", "--experiment", mil_project["exp"]) == 0
    assert "test" not in requested  # training never reads the test split
    out = capsys.readouterr().out
    assert out.startswith("train_mil: experiment=synth_abmil")
    run = _only_run_dir(mil_project["outputs"])
    info = json.loads((run / "run_info.json").read_text())
    assert info["status"] == "finished" and info["run_id"] == f"synth_abmil/{run.parent.name}/seed0"
    assert info["config_hash"].startswith(run.parent.name) and info["data"]["train_volumes"] == 160
    val = json.loads((run / "val_metrics.json").read_text())
    assert val["macro_auroc"] > 0.9, val  # the planted signal is learnable
    assert json.loads((run / "labels.json").read_text()) == LABELS
    metrics = pd.read_csv(run / "metrics.csv")
    assert list(metrics.epoch) == list(range(1, len(metrics) + 1)) and info["best_epoch"] in set(metrics.epoch)
    assert not list(run.glob("eval/*"))  # no test results from training
    ledger = pd.read_csv(mil_project["outputs"] / "index.csv")
    assert len(ledger) == 1 and ledger.run_id[0] == info["run_id"] and ledger.aggregator[0].startswith("abmil(")

    # a finished run is never overwritten; another seed is a new folder of the same config hash
    assert call(monkeypatch, "train_mil", "--experiment", mil_project["exp"]) == 1
    assert "already trained" in capsys.readouterr().err

    # test: once, with val thresholds, per kernel and per scanner, with CIs
    assert call(monkeypatch, "evaluate_mil", "--experiment-dir", run, "--bootstrap", "50", "--min-group", "30") == 0
    ev_dir = run / "eval" / "ctrate__synth__test"
    summary = pd.read_csv(ev_dir / "metrics_summary.csv")
    groups = set(zip(summary.group_by, summary.group))
    assert groups == {("all", "all"), ("kernel_class", "sharp"), ("kernel_class", "soft"),
                      ("manufacturer", "Philips"), ("manufacturer", "Siemens")}
    overall = summary[summary.group == "all"].iloc[0]
    assert overall.n_volumes == 80 and overall.macro_auroc_ci_low <= overall.macro_auroc <= overall.macro_auroc_ci_high
    per_label = pd.read_csv(ev_dir / "metrics_per_label.csv")
    assert set(zip(per_label.group_by, per_label.group, per_label.label)) == {
        (gb, g, label) for gb, g in groups for label in LABELS}  # every group has every label
    assert len(per_label) == len(groups) * len(LABELS)
    thresholds = json.loads((run / "thresholds.json").read_text())
    assert np.allclose(per_label[per_label.group == "all"].threshold, [thresholds[n] for n in LABELS])
    assert call(monkeypatch, "evaluate_mil", "--experiment-dir", run) == 1  # never silently redone
    evals = pd.read_csv(mil_project["outputs"] / "evaluations.csv")
    assert len(evals) == 1 and {"macro_auroc[sharp]", "macro_auroc[soft]"} <= set(evals.columns)

    capsys.readouterr()
    assert call(monkeypatch, "summarize_experiments", "--output-dir", mil_project["outputs"], "--split", "test") == 0
    assert "synth_abmil" in capsys.readouterr().out

    # L1 evidence: for positive val volumes, the top slice is one of the planted ones
    from ct_model.explain.volume import open_explainer

    ex = open_explainer(run, device="cpu")
    hits, total = 0, 0
    for vid in ex.volume_ids("val"):
        rec = ex.record(vid)
        ev = ex.evidence(vid)
        for c, name in enumerate(LABELS):
            if (rec.patient_id, c) in mil_project["planted"]:
                total += 1
                hits += int(ev.top_slices(name, 1)[0] in mil_project["planted"][(rec.patient_id, c)])
    assert total > 20 and hits / total >= 0.9, (hits, total)


def test_resume_continues_an_interrupted_run(mil_project, monkeypatch):
    from ct_model.training.config import load_experiment, override
    from ct_model.training.trainer import train_experiment

    cfg = override(load_experiment(str(mil_project["exp"])), **{"optim.max_epochs": 6, "optim.patience": 100})

    def crash_after_epoch_3(line):
        if line.strip().startswith("epoch   3"):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train_experiment(cfg, log=crash_after_epoch_3)
    run = _only_run_dir(mil_project["outputs"])
    assert json.loads((run / "run_info.json").read_text())["status"] == "running"
    logs = []
    result = train_experiment(cfg, log=logs.append)
    assert any("resuming" in line and "after epoch 3" in line for line in logs)
    assert result.epochs_run == 6
    assert list(pd.read_csv(run / "metrics.csv").epoch) == [1, 2, 3, 4, 5, 6]
