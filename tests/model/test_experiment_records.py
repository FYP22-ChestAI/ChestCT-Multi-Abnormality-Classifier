"""Experiment configs and records: strict loading, overrides, the config hash, the ledgers."""
from pathlib import Path

import pandas as pd
import pytest

from ct_model.training.config import load_experiment, override, parse_experiment
from ct_model.training.records import append_row, config_hash, run_dir, run_id

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "configs" / "model" / "experiments"


def test_repo_experiment_configs_load():
    for path in EXPERIMENTS.glob("*.yaml"):
        cfg = load_experiment(str(path))
        assert cfg.name == path.stem
    cfg = load_experiment("abmil_dale2s", EXPERIMENTS)
    assert cfg.data.run == "train-sharp-5800" and cfg.data.train_kernel_classes == ("sharp",)
    assert cfg.aggregator["type"] == "abmil" and cfg.optim.max_epochs == 100 and cfg.optim.batch_size == 16


def test_unknown_keys_and_bad_values_raise():
    with pytest.raises(ValueError, match="unexpected keyword"):
        parse_experiment({"name": "x", "optim": {"learning_rate": 1e-3}})
    with pytest.raises(ValueError, match="monitor"):
        parse_experiment({"name": "x", "optim": {"monitor": "accuracy"}})
    with pytest.raises(ValueError, match="slice sampler"):
        parse_experiment({"name": "x", "data": {"train_slice_sampler": {"mode": "best"}}})
    with pytest.raises(KeyError, match="optim.lrr"):
        override(load_experiment("abmil_dale2s", EXPERIMENTS), **{"optim.lrr": 1e-3})


def test_overrides():
    cfg = override(load_experiment("abmil_dale2s", EXPERIMENTS), **{"optim.lr": 1e-4, "loss": {"type": "asl"}, "seed": None})
    assert cfg.optim.lr == 1e-4 and cfg.loss == {"type": "asl"} and cfg.seed == 0


def test_config_hash_groups_seeds_and_separates_settings():
    base = load_experiment("abmil_dale2s", EXPERIMENTS)
    h = config_hash(base, "dale_ct_2s-4f144ab5", "m" * 64)
    same = [override(base, seed=3), override(base, device="cpu"), override(base, name="renamed"),
            override(base, output_dir="elsewhere")]
    assert all(config_hash(c, "dale_ct_2s-4f144ab5", "m" * 64) == h for c in same)
    different = [override(base, **{"optim.lr": 1e-4}), override(base, loss={"type": "asl"}),
                 override(base, aggregator={**base.aggregator, "branches": "per_label"})]
    assert len({config_hash(c, "dale_ct_2s-4f144ab5", "m" * 64) for c in different} | {h}) == 4
    assert config_hash(base, "other_store-00000000", "m" * 64) != h  # another encoder / weights / HU cache
    assert config_hash(base, "dale_ct_2s-4f144ab5", "n" * 64) != h   # another manifest (volumes, splits, labels)
    seed3 = override(base, seed=3)
    assert run_id(seed3, h) == f"abmil_dale2s/{h[:8]}/seed3"
    assert run_dir(seed3, h) == Path("outputs/experiments/abmil_dale2s") / h[:8] / "seed3"


def test_ledger_appends_and_grows_columns(tmp_path):
    ledger = tmp_path / "index.csv"
    append_row(ledger, {"run_id": "a", "val_macro_auroc": 0.8})
    append_row(ledger, {"run_id": "b", "val_macro_auroc": 0.81, "macro_auroc[soft]": 0.7})
    frame = pd.read_csv(ledger)
    assert list(frame.run_id) == ["a", "b"] and pd.isna(frame["macro_auroc[soft]"][0])
