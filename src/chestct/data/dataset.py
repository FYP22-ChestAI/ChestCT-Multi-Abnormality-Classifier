"""Step 10: the actual M1 -> M3 / M4 hand-off.

Nobody downstream opens a .npy by hand. ``ChestCTDataset`` reads the cached
.npy + manifest row and returns a ready-to-use tensor:

    .npy (int16 HU, N slices)
      -> pick slices -> windowing (windows.py) -> (K, 3, H, W) float32 in [0,1]
      -> optional ImageNet normalisation
      -> torch tensor

Two modes:
  - "all_lowres": every slice, downsampled -- for M3's Stage A scoring pass.
  - "selected": only specific slice indices, at full resolution -- for M4.
    M3 hasn't been built yet, so for now this mode expects the chosen indices
    to already be a column on the manifest (see docs/data_contract.md); until
    that exists, use mode="all_lowres" or call slices_to_tensor() directly.

This module duck-types as a PyTorch map-style Dataset (implements __len__ and
__getitem__), so it plugs into torch.utils.data.DataLoader without requiring
torch just to construct or inspect it -- torch is only imported inside
__getitem__, when a tensor is actually produced. Everything else in this
package works on a machine with no torch installed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .windows import DEFAULT_WINDOWS, apply_windows

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_slices(npy_path, indices: np.ndarray | None = None) -> np.ndarray:
    """Read chosen slices from a cached .npy without loading the whole volume."""
    arr = np.load(npy_path, mmap_mode="r")
    if indices is None:
        return np.asarray(arr)
    return np.asarray(arr[indices])


def slices_to_tensor(
    hu_slices: np.ndarray,
    windows: dict | None = None,
    normalize_imagenet: bool = False,
) -> np.ndarray:
    """(K, H, W) HU -> (K, 3, H, W) float32 in [0, 1], windowed, optionally
    ImageNet-normalised. This is the exact function a frozen 2D encoder's
    input is built from."""
    x = apply_windows(hu_slices, windows or DEFAULT_WINDOWS)
    if normalize_imagenet:
        mean = IMAGENET_MEAN.reshape(1, 3, 1, 1)
        std = IMAGENET_STD.reshape(1, 3, 1, 1)
        x = (x - mean) / std
    return x


class ChestCTDataset:
    def __init__(
        self,
        manifest: pd.DataFrame,
        cache_dir: str | Path,
        label_cols: list[str] | None = None,
        mask_cols: list[str] | None = None,
        mode: str = "all_lowres",
        low_res_size: int = 112,
        normalize_imagenet: bool = True,
        windows: dict | None = None,
        indices_col: str = "selected_indices",
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.cache_dir = Path(cache_dir)
        self.label_cols = label_cols
        self.mask_cols = mask_cols
        self.mode = mode
        self.low_res_size = low_res_size
        self.normalize_imagenet = normalize_imagenet
        self.windows = windows
        self.indices_col = indices_col

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, i: int):
        import torch  # imported here, not at module level -- see module docstring

        row = self.manifest.iloc[i]
        npy_path = self.cache_dir / f"{row['volume_id']}.npy"

        if self.mode == "all_lowres":
            hu = _load_slices(npy_path)  # need every slice, so no partial read here
            from scipy.ndimage import zoom

            factor = self.low_res_size / hu.shape[-1]
            hu = zoom(hu, (1.0, factor, factor), order=1)
            indices = np.arange(hu.shape[0])
        elif self.mode == "selected":
            if self.indices_col not in row or pd.isna(row[self.indices_col]):
                raise KeyError(
                    f"manifest has no '{self.indices_col}' for {row['volume_id']!r}. "
                    "mode='selected' expects M3's chosen slice indices already written "
                    "into the manifest (see docs/data_contract.md); use mode='all_lowres' "
                    "for M3's own scoring pass, or write that column yourself."
                )
            indices = np.array(json.loads(row[self.indices_col]))
            hu = _load_slices(npy_path, indices)
        else:
            raise ValueError(f"unknown mode: {self.mode!r} (expected 'all_lowres' or 'selected')")

        x = slices_to_tensor(hu, windows=self.windows, normalize_imagenet=self.normalize_imagenet)

        item = {
            "pixel_values": torch.from_numpy(np.ascontiguousarray(x)),
            "volume_id": row["volume_id"],
            "patient_id": row["patient_id"],
            "slice_indices": torch.from_numpy(np.ascontiguousarray(indices)),
        }

        # label_cols is None for inference use (a real scan has no label --
        # producing one is the model's job, not this Dataset's -- see
        # docs/data_contract.md). Training passes label_cols explicitly.
        if self.label_cols:
            labels = row[self.label_cols].to_numpy(dtype=np.float32)
            mask = row[self.mask_cols].to_numpy(dtype=np.float32) if self.mask_cols else np.ones_like(labels)
            item["labels"] = torch.from_numpy(labels)
            item["label_mask"] = torch.from_numpy(mask)

        return item
