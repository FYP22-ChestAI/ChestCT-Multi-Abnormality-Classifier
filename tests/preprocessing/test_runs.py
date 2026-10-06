"""Runs: named folders that coexist, are never overwritten, and share one cache."""
import json

import pytest

from ct_preprocessing.config import PathsConfig
from ct_preprocessing.runs import (
    DEFAULT_ARCHIVE_RUN, Run, auto_run_name, check_run_name, get_run, list_runs, new_run, read_run_info,
    resolve_run, write_run_info,
)


@pytest.fixture
def paths(tmp_path):
    return PathsConfig(runs_dir=str(tmp_path / "runs"), cache_dir=str(tmp_path / "cache"))


def test_a_run_has_its_own_folder_with_every_derived_file(paths):
    run = Run(paths, "ctrate", "train-sharp")
    assert run.dir.parts[-3:] == ("runs", "ctrate", "train-sharp")
    for path in (run.worklist_path, run.manifest_path, run.qc_report_path, run.montage_dir, run.record_path,
                 run.splits_copy_path, run.state_dir, run.chunk_manifest_dir, run.info_path):
        assert run.dir in path.parents
    assert run.chunk_manifest_path("0007").name == "chunk_0007.csv"


def test_two_runs_never_share_a_file(paths):
    a, b = Run(paths, "ctrate", "train-sharp"), Run(paths, "ctrate", "train-soft")
    assert a.manifest_path != b.manifest_path and a.worklist_path != b.worklist_path and a.state_dir != b.state_dir


def test_a_new_run_is_created_and_an_existing_one_is_never_touched(paths):
    run = new_run(paths, "ctrate", "train-sharp")
    (run.dir / "worklist.csv").write_text("precious")
    with pytest.raises(FileExistsError, match="never overwritten"):
        new_run(paths, "ctrate", "train-sharp")
    assert (run.dir / "worklist.csv").read_text() == "precious"


def test_run_names_are_checked():
    assert check_run_name("train-sharp") == "train-sharp" and check_run_name("a_1.2") == "a_1.2"
    for bad in ("", "../x", "a b", "-lead", "x/y"):
        with pytest.raises(ValueError, match="invalid run name"):
            check_run_name(bad)


def test_the_default_name_says_what_the_run_trains_on():
    assert auto_run_name("sharp") == "train-sharp" and auto_run_name("soft") == "train-soft"


def test_runs_are_listed_per_source(paths):
    assert list_runs(paths, "ctrate") == []
    new_run(paths, "ctrate", "train-soft")
    new_run(paths, "ctrate", "train-sharp")
    new_run(paths, "nhrd_local", "main")
    assert list_runs(paths, "ctrate") == ["train-sharp", "train-soft"] and list_runs(paths, "nhrd_local") == ["main"]


def test_resolving_a_run_by_name_or_as_the_only_one(paths):
    new_run(paths, "ctrate", "train-sharp")
    assert resolve_run(paths, "ctrate").name == "train-sharp"  # the only run
    assert resolve_run(paths, "ctrate", "train-sharp").name == "train-sharp"


def test_with_several_runs_a_name_is_required_and_the_choices_are_listed(paths):
    new_run(paths, "ctrate", "train-sharp")
    new_run(paths, "ctrate", "train-soft")
    with pytest.raises(ValueError, match="several runs \\(train-sharp, train-soft\\)"):
        resolve_run(paths, "ctrate")


def test_an_unknown_run_lists_the_existing_ones(paths):
    new_run(paths, "ctrate", "train-sharp")
    with pytest.raises(FileNotFoundError, match="no run 'nope'.*train-sharp"):
        get_run(paths, "ctrate", "nope")


def test_with_no_runs_the_message_points_at_make_worklist(paths):
    with pytest.raises(FileNotFoundError, match="make_worklist.py"):
        resolve_run(paths, "ctrate")


def test_an_archive_source_gets_its_default_run_created_on_first_use(paths):
    run = resolve_run(paths, "nhrd_local", create_default=DEFAULT_ARCHIVE_RUN)
    assert run.name == "main" and run.dir.is_dir()
    assert resolve_run(paths, "nhrd_local", create_default=DEFAULT_ARCHIVE_RUN).name == "main"  # and then it is simply found


def test_run_info_records_what_made_the_run(paths):
    run = new_run(paths, "ctrate", "train-soft")
    write_run_info(run, {"parent_run": "train-sharp", "settings": {"train_kernel": "soft"}})
    info = read_run_info(run)
    assert info["name"] == "train-soft" and info["source"] == "ctrate" and info["parent_run"] == "train-sharp"
    assert info["settings"] == {"train_kernel": "soft"} and "created" in info
    assert json.loads(run.info_path.read_text())["name"] == "train-soft"
    assert read_run_info(Run(paths, "ctrate", "missing")) == {}
