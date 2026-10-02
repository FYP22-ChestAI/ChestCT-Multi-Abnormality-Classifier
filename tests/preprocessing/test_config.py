"""The config file: defaults, nesting, and loud failures for typos."""
from pathlib import Path

import pytest

from ct_preprocessing.cli import pick
from ct_preprocessing.config import IngestConfig, SourceConfig, load_config

REPO = Path(__file__).resolve().parents[2]


def _write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_the_shipped_config_loads_with_the_documented_defaults():
    cfg = load_config(REPO / "configs" / "preprocessing.yaml")

    ctrate, nhrd = cfg.source("ctrate"), cfg.source("nhrd_local")
    assert ctrate.manifest_builder == "ctrate" and nhrd.manifest_builder == "folder"
    assert ctrate.ingest.chunk_size == 40
    assert ctrate.ingest.train_pool == "one_per_scan" and ctrate.ingest.test_pool == "all"
    assert ctrate.split.n_val_patients == 1000
    assert nhrd.split.n_val_patients == 150 and nhrd.split.n_test_patients == 150
    assert ctrate.ingest.min_free_gb == nhrd.ingest.min_free_gb == 100
    assert nhrd.ingest.drive_remote
    # the scratch folders are separate per source, and never the cache folder
    assert ctrate.raw_dir != nhrd.raw_dir
    assert cfg.paths.cache_dir not in (ctrate.raw_dir, nhrd.raw_dir)


def test_the_shipped_config_has_no_windowing_settings():
    # windowing belongs to the model code now; the cache is raw HU
    text = (REPO / "configs" / "preprocessing.yaml").read_text(encoding="utf-8")
    assert not any(line.startswith("windows:") for line in text.splitlines())
    assert not hasattr(load_config(REPO / "configs" / "preprocessing.yaml"), "windows")


def test_missing_keys_fall_back_to_dataclass_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, "sources:\n  s:\n    format: dicom\n    manifest_builder: folder\n"))
    s = cfg.source("s")
    assert s.ingest == IngestConfig()  # nothing configured -> the documented defaults
    assert s.patient_id_source == "auto" and s.split.seed == 0


def test_nested_ingest_and_split_blocks_are_parsed(tmp_path):
    cfg = load_config(_write(tmp_path, """
sources:
  s:
    format: dicom
    manifest_builder: folder
    ingest: {chunk_size: 7, drive_remote: "gdrive:x", min_free_gb: 5}
    split: {n_val_patients: 3, n_test_patients: 2}
"""))
    s = cfg.source("s")
    assert (s.ingest.chunk_size, s.ingest.drive_remote, s.ingest.min_free_gb) == (7, "gdrive:x", 5)
    assert (s.split.n_val_patients, s.split.n_test_patients) == (3, 2)


@pytest.mark.parametrize("bad", [
    "    format: nifti\n    chunks_size: 5\n",            # unknown key: a typo must not pass silently
    "    format: nifti\n    ingest: {chunk_size: 0}\n",   # invalid value
    "    format: tiff\n",                                  # not a supported format
    "    ingest: {train_pool: some}\n",
])
def test_typos_and_invalid_values_fail_loudly(tmp_path, bad):
    with pytest.raises(ValueError, match="source 's'|invalid config"):
        load_config(_write(tmp_path, "sources:\n  s:\n" + bad))


def test_asking_for_an_unknown_source_lists_the_known_ones(tmp_path):
    cfg = load_config(_write(tmp_path, "sources:\n  alpha: {}\n"))
    with pytest.raises(KeyError, match="alpha"):
        cfg.source("beta")


def test_source_config_validates_directly():
    with pytest.raises(ValueError):
        SourceConfig(patient_id_source="guess")


def test_pick_prefers_the_command_line_but_keeps_zero_and_false():
    assert pick(None, 40) == 40
    assert pick(5, 40) == 5
    assert pick(0, 40) == 0  # an explicit 0 is a value, not "unset"
