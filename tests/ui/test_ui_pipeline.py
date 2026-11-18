"""The control panel's one-click pipeline: status checks read saved outputs, the plan fans out and honours
modes, and the runner skips finished steps and stops at the first failure."""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app import pipeline, pipeline_runner, status  # noqa: E402
from app.pipeline import ALWAYS, SKIP, Stage, Target, build_steps, will_run  # noqa: E402


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(REPO)


@pytest.fixture
def project(tmp_path):
    """A preprocessing config whose folders live in tmp_path, and the run folder it implies."""
    raw = yaml.safe_load((REPO / "configs" / "preprocessing.yaml").read_text())
    raw["paths"] = {k: str(tmp_path / Path(v).name) if k != "kernel_table" else v for k, v in raw["paths"].items()}
    cfg = tmp_path / "preprocessing.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    run_dir = tmp_path / "runs" / "ctrate" / "r1"
    return {"cfg": str(cfg), "run": run_dir, "argv": ["--config", str(cfg), "--source", "ctrate", "--run", "r1"]}


def _worklist(run_dir: Path, chunks: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"chunk": list(range(chunks)), "volume_id": [f"v{i}" for i in range(chunks)]}).to_csv(
        run_dir / "worklist.csv", index=False)


def _done(run_dir: Path, *chunks: int):
    (run_dir / "state").mkdir(parents=True, exist_ok=True)
    for c in chunks:
        (run_dir / "state" / f"chunk_{c:04d}.done").write_text("{}")


def test_preprocessing_checks_follow_the_saved_outputs(project):
    run, argv = project["run"], project["argv"]
    assert status.worklist(["--config", project["cfg"], "--name", "r1"]).state == status.TODO
    assert status.ingest(argv).state == status.BLOCKED  # no run folder yet

    _worklist(run, 3)
    assert status.worklist(["--config", project["cfg"], "--name", "r1"]).state == status.DONE
    _done(run, 0)
    s = status.ingest(argv)
    assert s.state == status.PARTIAL and "1 / 3" in s.detail
    _done(run, 1, 2)
    assert status.ingest(argv).state == status.DONE

    assert status.merge(argv).state == status.TODO  # no manifest yet
    pd.DataFrame({"volume_id": ["v0", "v1", "v2"], "split": ["train", "unassigned", "test"]}).to_csv(
        run / "manifest.csv", index=False)
    assert status.merge(argv).state == status.DONE
    assert status.qc(argv).state == status.TODO
    pd.DataFrame({"volume_id": ["v0", "v1"], "passed": [True, False]}).to_csv(run / "qc_report.csv", index=False)
    assert "1 volumes not checked" in status.qc(argv).detail
    pd.DataFrame({"volume_id": ["v0", "v1", "v2"], "passed": [True, False, True]}).to_csv(run / "qc_report.csv", index=False)
    assert status.qc(argv).state == status.DONE
    assert status.splits(argv).state == status.TODO
    pd.DataFrame({"volume_id": ["v0", "v1", "v2"], "split": ["train", "excluded", "test"]}).to_csv(
        run / "manifest.csv", index=False)
    assert status.splits(argv).state == status.DONE


def test_a_chunk_finished_after_the_merge_makes_merge_due_again(project):
    import os, time
    run, argv = project["run"], project["argv"]
    _worklist(run, 2)
    _done(run, 0)
    pd.DataFrame({"volume_id": ["v0"], "split": ["train"]}).to_csv(run / "manifest.csv", index=False)
    old = time.time() - 100
    os.utime(run / "manifest.csv", (old, old))
    assert status.merge(argv).state == status.TODO


def _fake_stages(checks: dict):
    """Stages over a cheap script, with statuses chosen by the test."""
    def stage(key, shared=True, fan=None, always=None):
        expand = fan or (lambda t, make: [("", "", {"split": "val"})])
        return Stage(key, key.title(), "summarize-experiments", lambda argv, k=key: checks[k], expand,
                     shared=shared, always_args=always)
    per_seed = lambda t, make: [(f"{e}:{s}", f"{e} {s}", {"split": "val", "name": e}) for e in t.experiments for s in t.seeds]
    return [stage("prep"), stage("train", shared=False, fan=per_seed), stage("eval", shared=False, always={})]


def test_the_plan_fans_out_and_honours_modes(monkeypatch):
    checks = {"prep": status.Status(status.TODO, ""), "train": status.Status(status.DONE, ""),
              "eval": status.Status(status.DONE, "")}
    monkeypatch.setattr(pipeline, "STAGES", _fake_stages(checks))
    steps = build_steps(Target(experiments=["a", "b"], seeds=[0, 1]))
    assert [s.key for s in steps] == ["prep", "train:a:0", "train:a:1", "train:b:0", "train:b:1", "eval"]
    assert "--name" in steps[1].args and "a" in steps[1].args
    assert [will_run(s, {}) for s in steps] == [True, False, False, False, False, False]
    assert all(s.recheck for s in steps[1:])  # an earlier shared step runs first
    assert will_run(steps[0], {"prep": SKIP}) is False
    assert will_run(steps[-1], {"eval": ALWAYS}) is True
    assert will_run(steps[1], {"train:a:0": ALWAYS}) is False  # training cannot be redone: no always_args


def test_the_runner_skips_done_steps_and_stops_at_the_first_failure(monkeypatch, tmp_path):
    checks = {"prep": status.Status(status.DONE, ""), "train": status.Status(status.TODO, ""),
              "eval": status.Status(status.TODO, "")}
    monkeypatch.setattr(pipeline, "STAGES", _fake_stages(checks))
    calls = []

    def fake_call(cmd, cwd=None):
        calls.append(cmd)
        return 1 if len(calls) == 2 else 0  # the second training step fails
    monkeypatch.setattr(pipeline_runner.subprocess, "call", fake_call)
    monkeypatch.setattr(pipeline_runner.signal, "signal", lambda *a: None)
    plan = tmp_path / "x.plan.json"
    plan.write_text(json.dumps({"target": {"experiments": ["a"], "seeds": [0, 1, 2]}, "page_values": {}, "modes": {}}))

    assert pipeline_runner.main(str(plan)) == 1
    progress = pipeline_runner.read_progress(plan)
    states = {k: v["state"] for k, v in progress["steps"].items()}
    assert progress["result"] == "failed" and len(calls) == 2
    assert states == {"prep": "skipped", "train:a:0": "done", "train:a:1": "failed", "train:a:2": "pending",
                      "eval": "pending"}


def test_clinical_report_check_follows_the_evaluated_seeds(tmp_path):
    """BLOCKED until a seed is evaluated; TODO until the ensemble report covers exactly the evaluated seeds
    (and every seed's own report exists); DONE after; TODO again when another seed gets evaluated."""
    import numpy as np
    from ct_model.clinical import clinical_report

    names = ["a", "b"]
    rng = np.random.default_rng(0)
    config_dir = tmp_path / "exp" / "aaaa1111"

    def seed(k, evaluated=True):
        folder = config_dir / f"seed{k}"
        folder.mkdir(parents=True)
        (folder / "labels.json").write_text(json.dumps(names))
        (folder / "config.yaml").write_text(yaml.safe_dump({"name": "exp", "seed": k, "output_dir": str(tmp_path),
                                                            "data": {"source": "ctrate", "run": "r1"}}))
        (folder / "run_info.json").write_text(json.dumps({"run_id": f"exp/aaaa1111/seed{k}", "status": "finished",
                                                          "config_hash": "aaaa1111bbbb2222"}))
        for split, where in (("val", folder / "predictions_val.csv"),
                             ("test", folder / "eval" / "ctrate__r1__test" / "predictions.csv")):
            if split == "test" and not evaluated:
                continue
            where.parent.mkdir(parents=True, exist_ok=True)
            y = (rng.random((60, 2)) < 0.4).astype(float)
            frame = pd.DataFrame({"volume_id": [f"v{i}" for i in range(60)], "patient_id": [f"v{i}" for i in range(60)],
                                  "kernel_class": "sharp"})
            for c, n in enumerate(names):
                frame[f"prob_{n}"], frame[f"true_{n}"] = np.clip(0.5 * y[:, c] + rng.random(60) * 0.6, 0, 1), y[:, c]
            frame.to_csv(where, index=False)

    argv = ["--experiment-dir", str(config_dir), "--split", "test", "--per-seed"]
    seed(0, evaluated=False)
    assert status.clinical(argv).state == status.BLOCKED
    shutil.rmtree(config_dir / "seed0")
    seed(0)
    assert status.clinical(argv).state == status.TODO
    clinical_report(config_dir, per_seed=True, log=lambda m: None)  # the settings the check compares with
    assert status.clinical(argv).state == status.DONE
    seed(1)
    assert status.clinical(argv).state == status.TODO
    assert any(s.key == "clinical" for s in pipeline.STAGES)
