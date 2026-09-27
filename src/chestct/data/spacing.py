"""Step 5 (spacing half): resample a Volume so 1 voxel = the same real-world
distance for every scan.

This changes physical pixel size (mm/voxel), not the final pixel count --
CT-RATE scans have different body sizes, so the pixel count after this step
is still different per scan. Forcing every scan to one pixel COUNT is a
separate, later job (see resize.py / Step 7).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from .device import resolve_device
from .types import Volume


def _resample_gpu(hu: np.ndarray, new_shape: tuple[int, int, int]) -> np.ndarray:
    """Trilinear resample on the GPU via torch. Much faster than the CPU/scipy
    path and, crucially, the large intermediate array lives in GPU VRAM
    rather than system RAM -- this is what avoided the OOM in a reference
    notebook doing the equivalent CPU-vs-GPU comparison on real CT-RATE data.
    """
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(hu)).to("cuda", dtype=torch.float32)[None, None]
    t = F.interpolate(t, size=tuple(int(s) for s in new_shape), mode="trilinear", align_corners=False)
    out = t[0, 0].to("cpu").numpy()
    del t
    torch.cuda.empty_cache()
    return out


def resample_to_spacing(
    volume: Volume,
    target_spacing_zyx: tuple[float, float, float],
    min_change_ratio: float = 0.05,
    device: str = "cpu",
) -> Volume:
    """Resample ``volume`` onto a grid with ``target_spacing_zyx`` mm/voxel.

    Axes already within ``min_change_ratio`` of the target are left alone, to
    avoid blurring a volume that's already close to the right spacing
    (resampling twice loses detail for no benefit).

    ``device="cpu"`` (default) always uses the scipy path below, which needs
    no GPU and no torch. ``device="auto"``/``"cuda"`` uses a GPU if one is
    actually available (via device.resolve_device), otherwise it silently
    falls back to the same scipy path -- so the same config works on a
    plain CPU-only server and a Colab GPU runtime alike.
    """
    cur = np.asarray(volume.spacing, dtype=float)
    tgt = np.asarray(target_spacing_zyx, dtype=float)
    factors = cur / tgt  # >1 means we need MORE voxels on that axis (upsample)

    if np.all(np.abs(factors - 1.0) < min_change_ratio):
        return volume

    old_shape = np.array(volume.hu.shape)
    new_shape = np.maximum(np.round(old_shape * factors).astype(int), 1)
    real_factors = new_shape / old_shape  # the factor scipy will actually apply

    resolved = resolve_device(device)
    if resolved == "cuda":
        resampled = _resample_gpu(volume.hu, tuple(new_shape))
    else:
        resampled = zoom(volume.hu, real_factors, order=1)  # trilinear
    new_spacing = tuple((cur / real_factors).tolist())

    return Volume(
        hu=resampled,
        spacing=new_spacing,
        orientation=volume.orientation,
        meta={**volume.meta, "spacing_before_resample": volume.spacing, "resample_device": resolved},
    )
