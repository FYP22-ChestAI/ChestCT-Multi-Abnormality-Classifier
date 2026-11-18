"""Phase-2 training items: SCANS, not volumes, so the two reconstructions of one scan meet in one step.

* A scan with a sharp and a soft volume (``patient_id`` + ``scan_id`` in the manifest) is a PAIR: both
  volumes are views of the same anatomy with the same labels. The slices of the two views are aligned
  along z: the offset in [-max_offset, max_offset] that best matches the two per-slice mean-HU profiles
  (the kernel changes texture, not the mean). About 97.5 % of CT-RATE pairs are within 2 slices.
* A scan with one volume is a SINGLE; with ``augment_prob`` it gets a second, kernel-simulating view of
  the same slices (blur a sharp scan, sharpen a soft one -- applied on the GPU, ct_model.adapt.augment).

Each item returns int16 HU slices (k per view, sampled with SliceSampler: random_k for training,
uniform_k for validation), the slice indices, the labels and whether slice-level consistency applies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..data.slices import SliceSampler
from ..data.volumes import VolumeRecord

KERNEL_ORDER = ("sharp", "soft")


@dataclass(frozen=True)
class ScanItem:
    scan: str                          # "<patient_id>/<scan_id>"
    views: tuple[VolumeRecord, ...]    # 1 volume, or (sharp, soft)


def scan_items(records: list[VolumeRecord], manifest: pd.DataFrame, pairs: bool = True) -> list[ScanItem]:
    """Group volume records by scan. Without ``pairs`` (or without scan ids) every volume is its own item."""
    if not pairs or "scan_id" not in manifest.columns:
        return [ScanItem(r.volume_id, (r,)) for r in records]
    scan_of = dict(zip(manifest.volume_id.astype(str), manifest.patient_id.astype(str) + "/" + manifest.scan_id.astype(str)))
    groups: dict[str, list[VolumeRecord]] = {}
    for r in records:
        groups.setdefault(scan_of.get(r.volume_id, r.volume_id), []).append(r)
    items = []
    for scan, recs in groups.items():
        by_kernel = {r.kernel_class: r for r in recs}
        if all(k in by_kernel for k in KERNEL_ORDER):
            items.append(ScanItem(scan, tuple(by_kernel[k] for k in KERNEL_ORDER)))
            rest = [r for r in recs if r not in items[-1].views]
        else:
            rest = recs
        items += [ScanItem(f"{scan}#{r.volume_id}", (r,)) for r in rest]  # anything else: on its own
    return items


def align_offset(profile_a: np.ndarray, profile_b: np.ndarray, max_offset: int) -> tuple[int, bool]:
    """(offset, ok): slice i of A matches slice i + offset of B. ``ok`` is False when the volumes differ by
    more than ``max_offset`` slices or overlap too little to tell."""
    na, nb = len(profile_a), len(profile_b)
    if abs(na - nb) > max_offset:
        return 0, False
    best, best_err = 0, np.inf
    for off in range(-max_offset, max_offset + 1):
        lo, hi = max(0, -off), min(na, nb - off)
        if hi - lo < max(8, min(na, nb) // 2):
            continue
        err = float(np.mean(np.abs(profile_a[lo:hi] - profile_b[lo + off:hi + off])))
        if err < best_err:
            best, best_err = off, err
    return best, bool(np.isfinite(best_err))


def _load(rec: VolumeRecord) -> np.ndarray:
    vol = np.load(rec.npy_path, mmap_mode="r")
    if vol.ndim != 3 or vol.dtype != np.int16:
        raise ValueError(f"{rec.volume_id}: expected an int16 (N, H, W) volume, got {vol.dtype} {vol.shape}")
    return vol


class AdaptDataset(Dataset):
    """One item per scan: {"scan", "views": [{volume_id, kernel, hu (k, H, W) int16, idx}], "labels" (C,),
    "aligned": bool, "augment": None | "blur" | "sharpen"}. Use with a list-returning collate."""

    def __init__(self, items: list[ScanItem], label_index: list[int], sampler: SliceSampler, *,
                 max_offset: int = 2, augment_prob: float = 0.0, seed: int = 0):
        self.items = list(items)
        self.label_index = list(label_index)
        self.sampler = sampler
        self.max_offset = max_offset
        self.augment_prob = augment_prob
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.items)

    def labels(self, item: ScanItem) -> np.ndarray:
        return np.asarray([item.views[0].labels[j] for j in self.label_index], dtype=np.float32)

    def __getitem__(self, i: int) -> dict:
        item = self.items[i]
        rng = np.random.default_rng((self.seed, self.epoch, i))
        first = item.views[0]
        vol_a = _load(first)
        idx_a = self.sampler(vol_a.shape[0], rng)
        views = [{"volume_id": first.volume_id, "kernel": first.kernel_class, "idx": idx_a, "hu": np.array(vol_a[idx_a])}]
        aligned, augment = False, None
        if len(item.views) == 2:
            second = item.views[1]
            vol_b = _load(second)
            off, aligned = align_offset(vol_a.mean(axis=(1, 2)), vol_b.mean(axis=(1, 2)), self.max_offset)
            idx_b = np.clip(idx_a + off, 0, vol_b.shape[0] - 1) if aligned else self.sampler(vol_b.shape[0], rng)
            views.append({"volume_id": second.volume_id, "kernel": second.kernel_class, "idx": idx_b,
                          "hu": np.array(vol_b[idx_b])})
        elif self.augment_prob and rng.random() < self.augment_prob:
            augment = "sharpen" if first.kernel_class == "soft" else "blur"
            aligned = True  # the same slices
        return {"scan": item.scan, "views": views, "labels": self.labels(item), "aligned": aligned, "augment": augment}


def collate_items(batch: list[dict]) -> list[dict]:
    return batch
