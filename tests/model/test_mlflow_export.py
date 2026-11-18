"""Export to MLflow: idempotent, short metric keys, clinical reports and weights in, patient-level files out."""
import numpy as np
import pytest

pytest.importorskip("mlflow")

from ct_model.clinical import ClinicalConfig, clinical_report  # noqa: E402
from ct_model.tracking.mlflow_export import eval_key, export_outputs  # noqa: E402

from test_clinical import _fake_run  # noqa: E402  (tests/model is on sys.path: pytest rootdir import mode)


def test_export_is_idempotent_and_keeps_patient_files_out(tmp_path, tmp_path_factory, monkeypatch):
    import mlflow
    from mlflow.tracking import MlflowClient

    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    rng = np.random.default_rng(0)
    y_val = (rng.random((200, 2)) < 0.3).astype(float)
    y_test = np.repeat((rng.random((60, 2)) < 0.3).astype(float), 2, axis=0)
    runs = [_fake_run(tmp_path, s, y_val, y_test, noise=0.7) for s in (0, 1)]
    for r in runs:
        (r / "checkpoints").mkdir()
        (r / "checkpoints" / "best.pt").write_bytes(b"weights")
        (r / "checkpoints" / "last.pt").write_bytes(b"resume state")
    clinical_report(runs[0].parent, config=ClinicalConfig(bootstrap=0, min_group=10), per_seed=True, log=lambda m: None)

    uri = f"file:{tmp_path_factory.mktemp('server') / 'mlruns'}"  # outside the folder being exported
    first = export_outputs(tmp_path, None, tracking_uri=uri, log=lambda m: None)
    assert len(first.exported) == 3 and not first.failed   # 2 seeds + the ensemble
    again = export_outputs(tmp_path, None, tracking_uri=uri, log=lambda m: None)
    assert not again.exported and len(again.skipped) == 3

    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name("exp")
    seed0 = client.search_runs([exp.experiment_id], filter_string="tags.mlflow.runName = 'aaaa1111/seed0'")[0]
    assert "clinical.test.report_fp_per_scan" in seed0.data.metrics  # trained and evaluated on r1: short keys
    files = []

    def walk(path=""):
        for a in client.list_artifacts(seed0.info.run_id, path):
            walk(a.path) if a.is_dir else files.append(a.path)
    walk()
    assert "weights/best.pt" in files and not any(f.endswith("last.pt") for f in files)
    assert not any(f.rsplit("/", 1)[-1] in ("predictions.csv", "predictions_val.csv", "predictions_clinical.csv",
                                            "per_scan.csv", "paired_scans.csv") for f in files)
    ens = client.search_runs([exp.experiment_id], filter_string="tags.`ct.ensemble` = 'True'")
    assert len(ens) == 1 and ens[0].data.tags["ct.members"] == "2"


def test_eval_key_is_short_for_the_training_run():
    cfg = {"data": {"source": "ctrate", "run": "a+b"}}
    assert eval_key("ctrate__a+b__test", cfg) == "test"
    assert eval_key("nhrd_local__main__test", cfg) == "nhrd_local.main.test"
