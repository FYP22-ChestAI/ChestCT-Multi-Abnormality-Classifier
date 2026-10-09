"""The Streamlit control panel's logic (app/): no Streamlit needed -- argument reading, the command it
builds, the config defaults it shows, and that every script is registered."""
import argparse
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.argspec import APPEND, FLAG, LIST, VALUE, arg_specs, command_args, missing_required  # noqa: E402
from app.commands import script_info  # noqa: E402
from app.registry import PHASES, SCRIPTS  # noqa: E402


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(REPO)  # configs are read relative to the repo root, as the scripts do


def test_every_script_is_registered_once_and_has_a_build_parser():
    registered = [s.script for s in SCRIPTS]
    on_disk = sorted(p.relative_to(REPO).as_posix() for p in (REPO / "scripts").glob("*/*.py"))
    assert sorted(registered) == on_disk, "add new scripts to app/registry.py"
    assert len({s.key for s in SCRIPTS}) == len(SCRIPTS)
    for spec in SCRIPTS:
        assert spec.phase in PHASES
        dests = {a.dest for a in script_info(spec.script)[1]}
        assert set(spec.essential) <= dests, f"{spec.key}: essential names an unknown field"
        assert set(spec.choices) <= dests, f"{spec.key}: choices names an unknown field"


def _parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--splits", nargs="+", choices=["train", "val", "test"])
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--note")
    return ap


def test_argparse_is_read_into_field_kinds():
    specs = {s.dest: s for s in arg_specs(_parser())}
    assert [specs[d].kind for d in ("experiment", "lr", "splits", "add", "dry_run")] == [VALUE, VALUE, LIST, APPEND, FLAG]
    assert specs["experiment"].required and specs["lr"].type is float
    assert specs["splits"].choices == ("train", "val", "test")


def test_the_command_has_required_fields_and_only_what_changed():
    specs = arg_specs(_parser())
    defaults = {"lr": 2e-4, "splits": ["train", "val", "test"]}
    untouched = {"experiment": "abmil", "lr": 0.0002, "splits": ["train", "val", "test"], "add": None, "dry_run": False}
    assert command_args(specs, untouched, defaults) == ["--experiment", "abmil"]
    changed = {**untouched, "lr": 1e-4, "splits": ["test"], "add": ["A:B:sharp", "C:D:soft"], "dry_run": True,
               "note": "two words"}
    assert command_args(specs, changed, defaults) == [
        "--experiment", "abmil", "--lr", "0.0001", "--splits", "test",
        "--add", "A:B:sharp", "--add", "C:D:soft", "--dry-run", "--note", "two words"]
    assert missing_required(specs, {"experiment": None}) == ["--experiment"]


def test_train_defaults_come_from_the_experiment_file():
    from app.defaults import train_defaults

    d = train_defaults({"experiment": "abmil_dale2s"})
    assert d["patience"].value == 15 and d["max_epochs"].value == 100 and d["lr"].value == pytest.approx(2e-4)
    assert "optim.patience" in d["patience"].origin


def test_stage2_and_preprocessing_defaults_come_from_their_configs():
    from app.defaults import ingest_defaults, stage2_defaults

    s2 = stage2_defaults({})
    assert s2["encoder"].value == "dale_ct_2s" and s2["splits"].value == ["train", "val", "test"]
    ing = ingest_defaults({"source": "ctrate"})
    assert ing["min_free_gb"].value == 100 and ing["workers"].value == 4
