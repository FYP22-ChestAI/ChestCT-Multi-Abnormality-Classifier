"""Step 7 (crop half): cut away the black background around the body.

Method:
  1. Build a 3D true/false mask: voxel is "tissue" if HU is above a threshold
     that sits well above open air (~-1000 HU) and well below soft tissue
     (~-100 to +40 HU and up). Computed once over the whole volume.
  2. Keep only the largest connected blob, so a stray artifact outside the
     body (e.g. part of the scanner table) can't stretch the box.
  3. The bounding box is the min/max coordinate where the mask is true, plus
     a small margin.
  4. Cut that box out of the ORIGINAL HU volume -- the mask is only used to
     find where to cut; values inside the box are kept exactly as they were.

Air in the middle (lungs, trachea, bowel gas) does not break this: the box
only needs tissue at the OUTER edges, and the lungs are always surrounded by
chest wall / ribs / spine / skin at the same slice. So even though lung air
reads "false" in the mask, the box is still set by the surrounding tissue,
and the lungs simply sit inside it at their true (low) HU values -- which is
exactly what we want, since lung detail is the diagnostic target.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label as cc_label

from .types import Volume


def foreground_bbox(
    hu: np.ndarray,
    threshold_hu: float = -500.0,
    margin: int = 4,
    keep_largest_component: bool = True,
    cc_downsample: int = 4,
) -> tuple[slice, slice, slice]:
    """Return the smallest (z, y, x) box containing all tissue voxels.

    The largest-connected-component search (which removes stray artefacts
    outside the body, e.g. part of the scanner table) is run on a
    downsampled copy of the mask, not the full-resolution array. This is
    the fix for a real OOM we hit on a large real CT-RATE volume: a full
    3D connected-component labelling pass over a 1024x1024x237 boolean
    array was a major memory spike. A coarser grid gives an equally valid
    box -- the margin already added below covers the extra slack -- at a
    small fraction of the memory and compute cost.
    """
    mask = hu > threshold_hu

    if not mask.any():
        # Nothing above threshold at all -- return the full volume rather
        # than an empty box; quality.py will flag this scan for review.
        return tuple(slice(0, s) for s in hu.shape)

    if keep_largest_component:
        step = max(1, int(cc_downsample))
        coarse = mask[::step, ::step, ::step]
        labeled, n_components = cc_label(coarse)
        if n_components > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # background label
            largest = int(np.argmax(sizes))
            coarse_mask = labeled == largest
            coords = np.array(np.nonzero(coarse_mask))
            mins = coords.min(axis=1) * step
            maxs = (coords.max(axis=1) + 1) * step
            box = []
            for axis, (lo, hi) in enumerate(zip(mins, maxs)):
                lo = max(0, int(lo) - margin)
                hi = min(hu.shape[axis], int(hi) + margin)
                box.append(slice(lo, hi))
            return tuple(box)
        # n_components <= 1: nothing stray to remove -- fall through to the
        # cheap full-resolution bounding box below, using the original mask.

    coords = np.array(np.nonzero(mask))
    mins = coords.min(axis=1)
    maxs = coords.max(axis=1) + 1  # exclusive upper bound

    box = []
    for axis, (lo, hi) in enumerate(zip(mins, maxs)):
        lo = max(0, int(lo) - margin)
        hi = min(hu.shape[axis], int(hi) + margin)
        box.append(slice(lo, hi))
    return tuple(box)


def crop_to_foreground(
    volume: Volume, threshold_hu: float = -500.0, margin: int = 4, cc_downsample: int = 4
) -> Volume:
    """Crop ``volume`` to its foreground bounding box.

    Cropping only decides WHERE to cut. The returned voxel values are the
    original HU values inside the box, unchanged -- nothing is zeroed or
    masked, including internal air (lungs, trachea).
    """
    box = foreground_bbox(volume.hu, threshold_hu=threshold_hu, margin=margin, cc_downsample=cc_downsample)
    cropped_hu = volume.hu[box]
    return Volume(
        hu=cropped_hu,
        spacing=volume.spacing,
        orientation=volume.orientation,
        meta={**volume.meta, "crop_box": [[s.start, s.stop] for s in box]},
    )
