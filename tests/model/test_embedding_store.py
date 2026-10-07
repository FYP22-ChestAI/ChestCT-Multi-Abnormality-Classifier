"""The embedding store: atomic writes, freshness, and refusing to mix differently made embeddings."""
import json

import numpy as np
import pytest

from ct_model.config import parse_encoder_config
from ct_model.embeddings.store import STORE_RECORD, StoreMismatch, open_store


def _cfg(**over):
    raw = {"name": "enc", "arch": "a", "embed_dim": 4, "input": {"transform": "clip_zscore", "clip_hu": [-1000, 1000], "mean_hu": 0, "std_hu": 1}}
    raw.update(over)
    return parse_encoder_config(raw)


def test_write_load_has_and_index(tmp_path):
    store = open_store(tmp_path, _cfg(), hu_cache_fingerprint="hu1", provenance={"weights_commit": "c1"})
    assert store.dir == tmp_path / _cfg().store_name and (store.dir / STORE_RECORD).is_file()
    emb = np.random.default_rng(0).normal(size=(7, 4)).astype(np.float32)
    store.write("v1", emb)
    assert store.has("v1") and store.has("v1", 7) and not store.has("v1", 8) and not store.has("v2")
    loaded = store.load("v1")
    assert loaded.dtype == np.float16 and loaded.shape == (7, 4)
    np.testing.assert_array_equal(loaded, emb.astype(np.float16))
    assert not list(store.dir.glob("*.tmp"))
    index = (store.write_index()).read_text().splitlines()
    assert index == ["volume_id,n_slices,embed_dim", "v1,7,4"]


def test_float16_storage_is_lossless_for_bfloat16_outputs(tmp_path):
    torch = pytest.importorskip("torch")
    store = open_store(tmp_path, _cfg(), hu_cache_fingerprint=None)
    bf16 = (torch.randn(50, 4) * 30).to(torch.bfloat16).float().numpy()  # what bf16 autocast produces
    store.write("v", bf16)
    np.testing.assert_array_equal(store.load("v").astype(np.float32), bf16)


def test_refuses_unstorable_values(tmp_path):
    store = open_store(tmp_path, _cfg(), hu_cache_fingerprint=None)
    with pytest.raises(ValueError, match="NaN"):
        store.write("v", np.array([[np.nan, 0, 0, 0]]))
    with pytest.raises(ValueError, match="float16 range"):
        store.write("v", np.array([[1e6, 0, 0, 0]]))
    with pytest.raises(ValueError, match="expected"):
        store.write("v", np.zeros((3, 5)))
    assert not store.has("v")


def test_refuses_to_mix(tmp_path):
    open_store(tmp_path, _cfg(), hu_cache_fingerprint="hu1", provenance={"weights_commit": "c1"})
    open_store(tmp_path, _cfg(), hu_cache_fingerprint="hu1", provenance={"weights_commit": "c1"})  # same: fine
    open_store(tmp_path, _cfg(), hu_cache_fingerprint="hu1", create=False)  # stage 3 opening it: fine
    with pytest.raises(StoreMismatch, match="HU cache"):
        open_store(tmp_path, _cfg(), hu_cache_fingerprint="hu2")
    with pytest.raises(StoreMismatch, match="weights commit"):
        open_store(tmp_path, _cfg(), hu_cache_fingerprint="hu1", provenance={"weights_commit": "c2"})
    # another config goes to another folder, never into this one
    other = open_store(tmp_path, _cfg(pooling="mean_patch"), hu_cache_fingerprint="hu1")
    assert other.dir != open_store(tmp_path, _cfg(), hu_cache_fingerprint="hu1").dir


def test_record_is_complete(tmp_path):
    store = open_store(tmp_path, _cfg(), hu_cache_fingerprint="hu1", embed_dim=4, provenance={"weights_commit": "c1"})
    record = json.loads((store.dir / STORE_RECORD).read_text())
    assert record["encoder_fingerprint"] == _cfg().fingerprint()
    assert record["hu_cache_fingerprint"] == "hu1" and record["embed_dim"] == 4 and record["store_dtype"] == "float16"
    assert record["encoder_config"]["input"]["clip_hu"] == [-1000.0, 1000.0]


def test_open_missing_store_for_reading(tmp_path):
    with pytest.raises(FileNotFoundError, match="encode_volumes"):
        open_store(tmp_path, _cfg(), hu_cache_fingerprint=None, create=False)
