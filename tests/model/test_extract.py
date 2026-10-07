"""The stage-2 loop end to end on CPU: slice order, resume, OOM back-off, per-volume failures, and the
datasets stage 3 will read the result with."""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ct_model.config import parse_encoder_config  # noqa: E402
from ct_model.data.datasets import EmbeddingBagDataset, collate_bags  # noqa: E402
from ct_model.data.volumes import select_volumes  # noqa: E402
from ct_model.embeddings.extract import _OOM, encode_slices, encode_volumes  # noqa: E402
from ct_model.embeddings.store import open_store  # noqa: E402
from ct_model.encoders.base import SliceEncoder  # noqa: E402
from ct_model.encoders.transforms import InputTransform  # noqa: E402

CPU = torch.device("cpu")


class FakeEncoder(SliceEncoder):
    """Embedding of a slice = [mean HU, number of slices in its batch, 0, 0], so tests can check which row is
    which slice and how the volume was batched. Raises OOM for batches larger than ``oom_above``."""

    def __init__(self, oom_above: int | None = None):
        cfg = parse_encoder_config({"name": "fake", "arch": "none", "embed_dim": 4, "input": {
            "transform": "clip_zscore", "clip_hu": [-3000, 3000], "mean_hu": 0.0, "std_hu": 1.0, "size_hw": [32, 32]}})
        super().__init__(cfg, InputTransform(cfg.input), 4)
        self.oom_above = oom_above
        self.batches: list[int] = []

    def embed(self, x):
        if self.oom_above is not None and x.shape[0] > self.oom_above:
            raise _OOM("fake out of memory")
        self.batches.append(x.shape[0])
        k = x.shape[0]
        return torch.stack([x.mean(dim=(1, 2, 3)), torch.full((k,), float(k)), torch.zeros(k), torch.zeros(k)], dim=1)


def _setup(mini_project):
    manifest = pd.read_csv(mini_project["run_dir"] / "manifest.csv")
    records = select_volumes(manifest, mini_project["cache"])
    enc = FakeEncoder()
    store = open_store(mini_project["embeddings"], enc.cfg, hu_cache_fingerprint="hu", embed_dim=4)
    return records, enc, store


def test_rows_follow_cache_slice_order(mini_project):
    records, enc, store = _setup(mini_project)
    summary = encode_volumes(records, enc, store, device=CPU, amp_dtype=None, slice_batch_size=4, num_workers=0)
    assert summary.encoded == len(records) and not summary.failed and summary.skipped == 0
    for rec in records:
        hu = np.load(rec.npy_path)
        emb = store.load(rec.volume_id).astype(np.float32)
        assert emb.shape == (rec.n_slices, 4)
        np.testing.assert_allclose(emb[:, 0], hu.reshape(len(hu), -1).mean(1), rtol=2e-3)
    assert max(enc.batches) == 4


def test_resume_skips_finished_volumes(mini_project):
    records, enc, store = _setup(mini_project)
    first = encode_volumes(records[:2], enc, store, device=CPU, amp_dtype=None, num_workers=0)
    again = encode_volumes(records, enc, store, device=CPU, amp_dtype=None, num_workers=0)
    assert first.encoded == 2 and again.skipped == 2 and again.encoded == len(records) - 2


def test_oom_halves_the_batch_and_keeps_it(mini_project):
    records, _, store = _setup(mini_project)
    enc = FakeEncoder(oom_above=3)
    logs = []
    summary = encode_volumes(records, enc, store, device=CPU, amp_dtype=None, slice_batch_size=16, num_workers=0, log=logs.append)
    assert summary.encoded == len(records) and summary.slice_batch_size == 2
    assert max(enc.batches) <= 3 and any("lowered to" in line for line in logs)


def test_oom_at_batch_one_is_a_failure_not_a_hang():
    with pytest.raises(_OOM):
        encode_slices(FakeEncoder(oom_above=0), torch.zeros(3, 32, 32), device=CPU, amp_dtype=None, slice_batch_size=4)


def test_bad_volume_is_recorded_and_the_rest_continue(mini_project):
    records, enc, store = _setup(mini_project)
    np.save(records[1].npy_path, np.zeros((3, 32, 32), dtype=np.int16))  # slice count no longer matches the manifest
    summary = encode_volumes(records, enc, store, device=CPU, amp_dtype=None, num_workers=0, log=lambda _: None)
    assert list(summary.failed) == [records[1].volume_id] and "manifest says" in summary.failed[records[1].volume_id]
    assert summary.encoded == len(records) - 1 and not store.has(records[1].volume_id)


def test_systematic_failure_aborts(mini_project):
    records, enc, store = _setup(mini_project)
    enc.embed = lambda x: torch.full((x.shape[0], 4), float("nan"))
    with pytest.raises(RuntimeError, match="all failed"):
        encode_volumes(records, enc, store, device=CPU, amp_dtype=None, num_workers=0, log=lambda _: None)


def test_low_disk_stops_cleanly(mini_project):
    records, enc, store = _setup(mini_project)
    summary = encode_volumes(records, enc, store, device=CPU, amp_dtype=None, num_workers=0, min_free_gb=1e12)
    assert summary.stopped and summary.encoded == 0


def test_bags_for_stage_3(mini_project):
    records, enc, store = _setup(mini_project)
    encode_volumes(records, enc, store, device=CPU, amp_dtype=None, num_workers=0)
    ds = EmbeddingBagDataset(records, store)
    batch = collate_bags([ds[0], ds[1]])
    n0, n1 = records[0].n_slices, records[1].n_slices
    assert batch["bags"].shape == (2, max(n0, n1), 4)
    assert batch["mask"].sum(1).tolist() == [n0, n1]
    assert batch["labels"].tolist()[1] == [0.0, 0.0] and batch["label_mask"].tolist()[1] == [True, False]
    store.path(records[2].volume_id).unlink()
    with pytest.raises(FileNotFoundError, match="encode_volumes"):
        EmbeddingBagDataset(records, store)
