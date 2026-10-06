"""Step 7 (resize half): force every scan to the same pixel count.

Spacing (spacing.py) fixes physical pixel size (mm/voxel) but leaves pixel
COUNT different per patient, because bodies are different physical sizes.
This step is what actually forces every scan to one fixed grid (e.g.
224x224), because the network needs a fixed input size. After this step a
pixel no longer means exactly the same mm across patients -- a known,
accepted trade-off (also used by CT-CLIP/AnyMC3D), not a bug.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from .device import resolve_device


def _resize_gpu(hu: np.ndarray, target_h: int, target_w: int, device: str = "cuda") -> np.ndarray:
    """Bilinear resize with torch, treating the slice axis as a batch dimension.
    Same rationale as spacing.py's GPU path: faster, and keeps the large array
    off system RAM. ``align_corners=True`` matches scipy's ``zoom`` (see
    spacing.py), so the CPU and GPU paths give the same array."""
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(hu)).to(device, dtype=torch.float32)[None]  # (1, Z, H, W)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=True)
    out = t[0].to("cpu").numpy()
    del t
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _force_exact_hw(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Guarantee the exact target shape, correcting for any off-by-one from
    floating-point zoom-factor rounding."""
    z, h, w = arr.shape
    if h == target_h and w == target_w:
        return arr
    out = np.full((z, target_h, target_w), fill_value=arr.min(), dtype=arr.dtype)
    copy_h, copy_w = min(h, target_h), min(w, target_w)
    out[:, :copy_h, :copy_w] = arr[:, :copy_h, :copy_w]
    return out


def resize_slices(
    hu: np.ndarray, target_hw: tuple[int, int], mode: str = "stretch", device: str = "cpu"
) -> np.ndarray:
    """Resize every axial slice (Y, X) of a (Z, Y, X) volume to a fixed (H, W).

    mode="stretch" (default): independent per-axis resize to target_hw.
        Simple; slightly distorts body shape if the crop box isn't square.
        Matches how most CT pipelines (CT-CLIP, AnyMC3D) handle this.
        Uses a GPU (torch) if device="auto"/"cuda" and one is available,
        else the scipy CPU path -- see spacing.py for the same pattern.
    mode="pad": resize preserving aspect ratio, then pad the shorter side
        with the volume's minimum value (background air) to reach a centred
        square of target_hw. No distortion, a few wasted border pixels.
        (CPU-only for now; this mode is used far less often.)
    """
    z, h, w = hu.shape
    target_h, target_w = target_hw

    if mode == "stretch":
        resolved = resolve_device(device)
        if resolved == "cuda":
            resized = _resize_gpu(hu, target_h, target_w)
        else:
            factors = (1.0, target_h / h, target_w / w)
            resized = zoom(hu, factors, order=1)
        return _force_exact_hw(resized.astype(hu.dtype), target_h, target_w)

    if mode == "pad":
        scale = min(target_h / h, target_w / w)
        new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
        resized = zoom(hu, (1.0, new_h / h, new_w / w), order=1).astype(hu.dtype)
        resized = _force_exact_hw(resized, min(new_h, target_h), min(new_w, target_w))

        out = np.full((z, target_h, target_w), fill_value=hu.min(), dtype=hu.dtype)
        top = (target_h - resized.shape[1]) // 2
        left = (target_w - resized.shape[2]) // 2
        out[:, top : top + resized.shape[1], left : left + resized.shape[2]] = resized
        return out

    raise ValueError(f"unknown resize mode: {mode!r} (expected 'stretch' or 'pad')")
