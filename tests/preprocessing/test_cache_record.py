"""The cache remembers its preprocessing settings, so a new run cannot silently overwrite what old runs use."""
import json

import pytest

from ct_preprocessing.cache_record import CACHE_RECORD, CacheMismatch, ensure_cache_matches, read_cache_fingerprint
from ct_preprocessing.preprocess import config_fingerprint
from ct_preprocessing.preprocess_config import PreprocessConfig

CFG = PreprocessConfig(target_size_hw=(32, 32))
OTHER = PreprocessConfig(target_size_hw=(64, 64))


def test_an_empty_cache_adopts_the_settings_it_is_first_used_with(tmp_path):
    ensure_cache_matches(tmp_path / "cache", CFG)
    assert read_cache_fingerprint(tmp_path / "cache") == config_fingerprint(CFG)
    record = json.loads((tmp_path / "cache" / CACHE_RECORD).read_text())
    assert record["preprocess_config"]["target_size_hw"] == [32, 32]


def test_the_same_settings_are_always_accepted(tmp_path):
    for _ in range(3):
        ensure_cache_matches(tmp_path, CFG)


def test_the_device_does_not_count_as_a_change(tmp_path):
    ensure_cache_matches(tmp_path, CFG)
    ensure_cache_matches(tmp_path, PreprocessConfig(target_size_hw=(32, 32), device="cuda"))


def test_other_settings_are_refused_before_anything_is_overwritten(tmp_path):
    ensure_cache_matches(tmp_path, CFG)
    with pytest.raises(CacheMismatch, match="existing runs point to"):
        ensure_cache_matches(tmp_path, OTHER)
    assert read_cache_fingerprint(tmp_path) == config_fingerprint(CFG)  # the record was not touched


def test_new_settings_can_be_allowed_explicitly(tmp_path):
    ensure_cache_matches(tmp_path, CFG)
    ensure_cache_matches(tmp_path, OTHER, allow_new_settings=True)
    assert read_cache_fingerprint(tmp_path) == config_fingerprint(OTHER)
    ensure_cache_matches(tmp_path, OTHER)  # and now they are the cache's own settings


def test_a_cache_from_before_records_existed_is_adopted_when_its_volumes_match(tmp_path):
    (tmp_path / "v1.meta.json").write_text(json.dumps({"fingerprint": config_fingerprint(CFG)}))
    ensure_cache_matches(tmp_path, CFG)
    assert read_cache_fingerprint(tmp_path) == config_fingerprint(CFG)


def test_a_cache_from_before_records_existed_is_refused_when_its_volumes_differ(tmp_path):
    (tmp_path / "v1.meta.json").write_text(json.dumps({"fingerprint": "0123456789abcdef"}))
    with pytest.raises(CacheMismatch, match="other preprocessing settings"):
        ensure_cache_matches(tmp_path, CFG)
    assert read_cache_fingerprint(tmp_path) is None  # nothing was recorded
    ensure_cache_matches(tmp_path, CFG, allow_new_settings=True)


def test_a_damaged_record_is_treated_as_missing_not_as_a_crash(tmp_path):
    (tmp_path / CACHE_RECORD).write_text("{not json")
    assert read_cache_fingerprint(tmp_path) is None
    ensure_cache_matches(tmp_path, CFG)
    assert read_cache_fingerprint(tmp_path) == config_fingerprint(CFG)
