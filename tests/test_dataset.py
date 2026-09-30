import numpy as np
import pandas as pd
import pytest

from chestct.preprocessing.dataset import slices_to_tensor


def test_slices_to_tensor_shape_and_range():
    hu = np.random.default_rng(0).uniform(-1000, 400, size=(5, 32, 32)).astype(np.float32)
    out = slices_to_tensor(hu, normalize_imagenet=False)
    assert out.shape == (5, 3, 32, 32)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_slices_to_tensor_imagenet_normalisation_shifts_mean():
    hu = np.zeros((2, 16, 16), dtype=np.float32)
    plain = slices_to_tensor(hu, normalize_imagenet=False)
    normed = slices_to_tensor(hu, normalize_imagenet=True)
    assert not np.allclose(plain, normed)


def test_chestct_dataset_getitem_requires_torch_or_is_skipped(tmp_path):
    torch = pytest.importorskip("torch")

    from chestct.preprocessing.dataset import ChestCTDataset

    npy_path = tmp_path / "train_1_a_1.npy"
    np.save(npy_path, np.zeros((10, 224, 224), dtype=np.int16))

    manifest = pd.DataFrame(
        {
            "volume_id": ["train_1_a_1"],
            "patient_id": ["train_1"],
            "label_a": [1.0],
            "label_b": [0.0],
        }
    )
    ds = ChestCTDataset(manifest, cache_dir=tmp_path, label_cols=["label_a", "label_b"], mode="all_lowres", low_res_size=112)

    item = ds[0]
    assert item["pixel_values"].shape == (10, 3, 112, 112)
    assert item["labels"].tolist() == [1.0, 0.0]


def test_chestct_dataset_with_no_labels_for_inference_use(tmp_path):
    """The exact same Dataset class, with label_cols omitted -- this is what
    lets a future inference entry point reuse it: a real scan has no label,
    producing one is the model's job (see docs/data_contract.md)."""
    pytest.importorskip("torch")
    from chestct.preprocessing.dataset import ChestCTDataset

    npy_path = tmp_path / "case_1.npy"
    np.save(npy_path, np.zeros((10, 224, 224), dtype=np.int16))

    manifest = pd.DataFrame({"volume_id": ["case_1"], "patient_id": ["case_1"]})
    ds = ChestCTDataset(manifest, cache_dir=tmp_path, mode="all_lowres", low_res_size=112)

    item = ds[0]
    assert "labels" not in item
    assert "label_mask" not in item
    assert item["pixel_values"].shape == (10, 3, 112, 112)
