"""PyTorch datasets over the HU cache (stage 2, and phase-2 LoRA training) and over the embedding
store (stages 3 + 4).

Memory: cache files are opened with ``mmap_mode="r"``, so only the slices asked for are read, and
slices stay int16 until they reach the device (half the bytes of float32 through the DataLoader and
the host-to-device copy). The intensity transform runs on the device, inside the encoder.

A volume that cannot be read does not crash a DataLoader worker: the item carries ``error`` instead,
so the caller records the failure and moves on (the same rule as ct_preprocessing's ingest).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .slices import SliceSampler
from .volumes import VolumeRecord


class HUVolumeDataset(Dataset):
    """One item per volume: its (K, H, W) int16 HU slices, chosen by ``sampler`` (default: all).

    Use with ``DataLoader(batch_size=None)``: volumes have different slice counts, and the encoder
    splits each one into slice batches itself.
    """

    def __init__(self, records: list[VolumeRecord], sampler: SliceSampler | None = None, seed: int = 0):
        self.records = list(records)
        self.sampler = sampler or SliceSampler("all")
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        item = {"index": i, "volume_id": rec.volume_id, "hu": None, "slice_idx": None, "error": None}
        try:
            volume = np.load(rec.npy_path, mmap_mode="r")
            if volume.ndim != 3 or volume.dtype != np.int16:
                raise ValueError(f"expected an int16 (N, H, W) volume, got {volume.dtype} {volume.shape}")
            if volume.shape[0] != rec.n_slices:
                raise ValueError(f"cache has {volume.shape[0]} slices, the manifest says {rec.n_slices}")
            idx = self.sampler(volume.shape[0], np.random.default_rng((self.seed, i)))
            # read into memory (a copy), so the tensor owns its data and the cache file is not held open
            slices = np.array(volume) if self.sampler.mode == "all" else volume[idx]
            item["hu"] = torch.from_numpy(slices)
            item["slice_idx"] = torch.from_numpy(idx)
        except Exception as exc:  # noqa: BLE001 - reported per volume, never fatal for the loader
            item["error"] = f"{type(exc).__name__}: {exc}"
        return item


class EmbeddingBagDataset(Dataset):
    """Stages 3 + 4: one item per volume -- its bag of slice embeddings and its label vector.

    ``store`` is a ct_model.embeddings.store.EmbeddingStore. Missing labels (NaN) are returned as 0
    with ``label_mask`` False, so the loss can ignore them.
    """

    def __init__(self, records: list[VolumeRecord], store, sampler: SliceSampler | None = None, seed: int = 0):
        missing = [r.volume_id for r in records if not store.has(r.volume_id, r.n_slices)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} volume(s) have no embeddings in {store.dir} (e.g. {missing[:5]}) -- "
                "run scripts/model/encode_volumes.py for this run first"
            )
        self.records = list(records)
        self.store = store
        self.sampler = sampler or SliceSampler("all")
        self.seed = seed
        self.epoch = 0  # set by the training loop so random_k draws new slices every epoch

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        emb = self.store.load(rec.volume_id)
        idx = self.sampler(emb.shape[0], np.random.default_rng((self.seed, self.epoch, i)))
        labels = np.asarray(rec.labels, dtype=np.float32)
        mask = ~np.isnan(labels)
        return {
            "volume_id": rec.volume_id,
            "bag": torch.from_numpy(np.asarray(emb[idx], dtype=np.float32)),
            "slice_idx": torch.from_numpy(idx),
            "labels": torch.from_numpy(np.nan_to_num(labels, nan=0.0)),
            "label_mask": torch.from_numpy(mask),
        }


def collate_bags(items: list[dict]) -> dict:
    """Pad bags of different lengths to (B, N_max, D) with a (B, N_max) bool mask (True = real slice)."""
    n_max = max(it["bag"].shape[0] for it in items)
    dim = items[0]["bag"].shape[1]
    bags = torch.zeros(len(items), n_max, dim)
    mask = torch.zeros(len(items), n_max, dtype=torch.bool)
    slice_idx = torch.full((len(items), n_max), -1, dtype=torch.long)
    for b, it in enumerate(items):
        n = it["bag"].shape[0]
        bags[b, :n] = it["bag"]
        mask[b, :n] = True
        slice_idx[b, :n] = it["slice_idx"]
    return {
        "volume_id": [it["volume_id"] for it in items],
        "bags": bags,
        "mask": mask,
        "slice_idx": slice_idx,
        "labels": torch.stack([it["labels"] for it in items]),
        "label_mask": torch.stack([it["label_mask"] for it in items]),
    }
