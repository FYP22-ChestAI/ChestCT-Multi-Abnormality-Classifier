"""Reading experiment folders back for reports: discovery, setup labels, per-finding tables."""
import json

import pandas as pd
import pytest
import yaml

from ct_model.training.results import load_results, mean_std, pm


def _run(root, name, cfg_hash, seed, loss="bce", max_epochs=100, aggregator="abmil", auroc=0.8, test=False):
    folder = root / name / cfg_hash[:8] / f"seed{seed}"
    folder.mkdir(parents=True)
    agg = {"type": aggregator, **({"attn_dim": 128} if aggregator == "abmil" else {})}
    cfg = {"name": name, "seed": seed, "device": "auto", "aggregator": agg, "loss": {"type": loss},
           "optim": {"max_epochs": max_epochs, "lr": 2e-4}}
    (folder / "config.yaml").write_text(yaml.safe_dump(cfg))
    (folder / "run_info.json").write_text(json.dumps({"run_id": f"{name}/{cfg_hash[:8]}/seed{seed}", "config_hash": cfg_hash,
                                                      "status": "finished", "best_epoch": 3, "epochs_run": 5}))
    (folder / "val_metrics.json").write_text(json.dumps({"n_volumes": 10, "macro_auroc": auroc, "val_loss": 0.3}))
    pd.DataFrame({"epoch": [1, 2], "lr": [1e-4, 2e-4], "val_macro_auroc": [0.7, auroc]}).to_csv(folder / "metrics.csv", index=False)
    pd.DataFrame({"label": ["a", "b"], "auroc": [auroc, auroc - 0.1]}).to_csv(folder / "val_per_label.csv", index=False)
    if test:
        ev = folder / "eval" / "ctrate__r__test"
        ev.mkdir(parents=True)
        pd.DataFrame({"group_by": ["all"], "group": ["all"], "macro_auroc": [auroc - 0.01]}).to_csv(ev / "metrics_summary.csv", index=False)
        pd.DataFrame({"group_by": ["all"], "group": ["all"], "label": ["a"], "auroc": [0.7]}).to_csv(ev / "metrics_per_label.csv", index=False)
    return folder


def test_load_results_finds_every_run_and_labels_configs(tmp_path):
    for seed, auroc in [(0, 0.80), (1, 0.82)]:
        _run(tmp_path, "abmil", "aaaa1111bbbb2222", seed, auroc=auroc, test=True)
    _run(tmp_path, "abmil", "cccc3333dddd4444", 0, loss="asl")
    _run(tmp_path, "abmil", "eeee5555ffff6666", 0, max_epochs=30)
    _run(tmp_path, "meanpool", "9999000011112222", 0, aggregator="mean_pool")

    res = load_results(tmp_path)
    assert len(res.runs) == 5 and set(res.runs.status) == {"finished"}
    setup = res.runs.drop_duplicates("hash8").set_index("hash8").setup
    # only departures from the most common value; attn_dim follows from the aggregator, so it is not listed
    assert setup.to_dict() == {"aaaa1111": "reference", "cccc3333": "loss=asl", "eeee5555": "max_epochs=30",
                               "99990000": "aggregator=mean_pool"}
    assert "val_loss" in res.runs and "val_val_loss" not in res.runs
    assert len(res.curves) == 10 and {"finding", "config", "setup"} <= set(res.val_labels.columns)
    assert len(res.test) == 2 and set(res.test.eval_split) == {"test"} and "finding" in res.test_labels

    table = mean_std(res.runs, ["hash8"], ["val_macro_auroc"])
    assert table.loc["aaaa1111", "seeds"] == 2 and table.loc["aaaa1111", "val_macro_auroc"] == pytest.approx(0.81)
    assert pm(table, ["val_macro_auroc"]).loc["aaaa1111", "val_macro_auroc"] == "0.8100 ± 0.0141"


def test_load_results_from_a_sub_folder_and_missing_root(tmp_path):
    folder = _run(tmp_path, "abmil", "aaaa1111bbbb2222", 0)
    assert load_results(folder).runs.run_id.tolist() == ["abmil/aaaa1111/seed0"]
    assert load_results(folder).runs.setup.tolist() == ["reference"]
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "nothing")
