"""Manifest rows -> volume records: the data contract's filters, path resolution and labels."""
import math

import pandas as pd
import pytest

from ct_model.data.slices import SliceSampler
from ct_model.data.volumes import label_names, missing_cache_files, select_volumes

import numpy as np


def _manifest(mini_project):
    return pd.read_csv(mini_project["run_dir"] / "manifest.csv")


def test_only_usable_splits_and_qc_passed(mini_project):
    recs = select_volumes(_manifest(mini_project), mini_project["cache"])
    assert [r.volume_id for r in recs] == ["train_1_a_1", "train_2_a_1", "train_3_a_1", "valid_1_a_1", "valid_1_a_2"]
    assert {r.split for r in recs} == {"train", "val", "test"}


def test_kernel_and_split_filters(mini_project):
    recs = select_volumes(_manifest(mini_project), mini_project["cache"], splits=("test",), kernel_classes=("soft",))
    assert [r.volume_id for r in recs] == ["valid_1_a_2"]
    with pytest.raises(ValueError, match="splits"):
        select_volumes(_manifest(mini_project), mini_project["cache"], splits=("excluded",))


def test_cache_path_ignores_manifest_npy_path(mini_project):
    recs = select_volumes(_manifest(mini_project), mini_project["cache"])
    assert all(r.npy_path == mini_project["cache"] / f"{r.volume_id}.npy" for r in recs)
    assert missing_cache_files(recs) == []
    (mini_project["cache"] / "train_1_a_1.npy").unlink()
    assert missing_cache_files(recs) == ["train_1_a_1"]


def test_labels_in_column_order_with_nan_for_missing(mini_project):
    m = _manifest(mini_project)
    assert label_names(m) == ["lung_nodule", "emphysema"]
    recs = {r.volume_id: r for r in select_volumes(m, mini_project["cache"])}
    assert recs["train_1_a_1"].labels == (1.0, 0.0)
    assert recs["train_2_a_1"].labels[0] == 0.0 and math.isnan(recs["train_2_a_1"].labels[1])


@pytest.mark.parametrize("n", [1, 5, 64, 230])
def test_slice_samplers_sorted_in_range_and_covering(n):
    rng = np.random.default_rng(0)
    assert list(SliceSampler("all")(n)) == list(range(n))
    assert list(SliceSampler("stride", stride=2)(n)) == list(range(0, n, 2))
    for mode in ("uniform_k", "random_k"):
        idx = SliceSampler(mode, k=32)(n, rng)
        assert len(idx) == min(n, 32) and len(set(idx)) == len(idx)
        assert list(idx) == sorted(idx) and idx.min() >= 0 and idx.max() < n
        if n > 32:  # one slice from each of k segments: head-to-foot coverage
            assert idx[0] < n / 32 and idx[-1] >= n - n / 32 - 1


def test_random_k_varies_uniform_k_does_not():
    a, b = SliceSampler("random_k", k=16)(200, np.random.default_rng(1)), SliceSampler("random_k", k=16)(200, np.random.default_rng(2))
    assert not np.array_equal(a, b)
    assert np.array_equal(SliceSampler("uniform_k", k=16)(200), SliceSampler("uniform_k", k=16)(200))
    with pytest.raises(ValueError):
        SliceSampler("random_k")
