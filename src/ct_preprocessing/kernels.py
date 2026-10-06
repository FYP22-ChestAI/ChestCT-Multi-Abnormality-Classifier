"""Reconstruction kernels: which volumes are "sharp" (lung / high-resolution) and which are "soft".

One CT scan is usually reconstructed twice from the same raw data -- a sharp kernel
(fine detail, more noise; used for lung findings) and a soft kernel (smooth, less
noise; used for mediastinum). CT-RATE keeps both as separate volumes (``_1``, ``_2``).
Which one a worklist takes is a setting (``train_kernel``), and the kernel and its
class are columns of every manifest, so later stages filter by plain pandas.

The class of a (manufacturer, kernel) pair comes from ``configs/kernel_classes.csv``,
a table that is reviewed by a person (see ``make_kernel_table.py`` and the kernel
survey). A pair that is not in the table is ``other`` and is never chosen as sharp or
soft -- only by a test pool that takes every reconstruction.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SHARP, SOFT, OTHER = "sharp", "soft", "other"
CLASSES = (SHARP, SOFT, OTHER)

_TOKEN = re.compile(r"[A-Za-z0-9_.+-]+")


def normalize_kernel(raw) -> str:
    """The kernel name from whatever the metadata holds: ``"['Br40f', '3']"`` -> ``"Br40f"``,
    ``"YA"`` -> ``"YA"``, empty / NaN -> ``""``. Only the first token counts (the second is
    the iterative-reconstruction strength on Siemens)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    tokens = _TOKEN.findall(str(raw))
    return tokens[0] if tokens else ""


def _key(manufacturer, kernel) -> tuple[str, str]:
    return (str(manufacturer or "").strip().lower(), normalize_kernel(kernel).lower())


def load_kernel_table(path: str | Path) -> dict[tuple[str, str], str]:
    """(manufacturer, kernel) -> class, from the reviewed CSV (columns ``manufacturer,kernel,class``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"kernel table {path} not found -- it lists which kernels are sharp and which are soft "
            "(draft one with scripts/preprocessing/make_kernel_table.py)"
        )
    df = pd.read_csv(path, dtype=str).fillna("")
    bad = sorted(set(df["class"]) - set(CLASSES))
    if bad:
        raise ValueError(f"{path}: class must be one of {list(CLASSES)}, got {bad}")
    return {_key(m, k): c for m, k, c in zip(df["manufacturer"], df["kernel"], df["class"])}


def classify(manufacturer, kernel, table: dict[tuple[str, str], str]) -> str:
    return table.get(_key(manufacturer, kernel), OTHER)


def add_kernel_columns(
    df: pd.DataFrame, table: dict[tuple[str, str], str], *, manufacturer_col: str, kernel_col: str
) -> pd.DataFrame:
    """``df`` with ``kernel`` (normalised name) and ``kernel_class`` (sharp / soft / other)."""
    out = df.copy()
    out["kernel"] = [normalize_kernel(k) for k in out[kernel_col]]
    out["kernel_class"] = [classify(m, k, table) for m, k in zip(out[manufacturer_col], out["kernel"])]
    return out


# ------------------------------------------------------------ drafting the table (a starting point only)
_LUNG_WORDS = ("HRCT", "PARANKIM", "LUNG")
_SOFT_WORDS = ("MEDIASTEN",)
_SIEMENS = ("siemens healthineers", "siemens")
_SHARP_LETTER_CODES = {"YA", "L", "YB", "EA", "LUNGB", "E"}
_SOFT_LETTER_CODES = {"B", "A", "SA"}


def draft_class(manufacturer, kernel, description="") -> str:
    """A first guess at one volume's class, from its series description and kernel name.

    Only used to DRAFT ``configs/kernel_classes.csv``; at run time the reviewed table decides.
    The description names the purpose (HRCT / parenchyma / lung vs mediastinum); Siemens kernel
    names encode sharpness (Bl = lung, Br / Bv / Qr number: up to 44 soft, 56 and up sharp,
    B31s soft, B70s sharp); Philips / PNMS letter codes are used only where descriptions confirm them.
    """
    desc = str(description or "").upper()
    name = normalize_kernel(kernel)
    if any(w in desc for w in _LUNG_WORDS):
        return SHARP
    if any(w in desc for w in _SOFT_WORDS):
        return SOFT
    if str(manufacturer or "").strip().lower() in _SIEMENS:
        if name.startswith(("Bl", "Hr")):
            return SHARP
        m = re.match(r"^(?:Br|Bv|Qr|Hc)(\d+)", name)
        if m:
            return SHARP if int(m.group(1)) >= 56 else SOFT
        m = re.match(r"^B(\d+)s", name)
        if m:
            return SHARP if int(m.group(1)) >= 60 else SOFT
    if name.upper() in _SHARP_LETTER_CODES:
        return SHARP
    if name.upper() in _SOFT_LETTER_CODES:
        return SOFT
    return OTHER


# ---------------------------------------------------------------------- the kernel survey
def sharpness_score(volume: np.ndarray, *, lo: float = -900.0, hi: float = 300.0) -> float:
    """How much fine detail and noise a volume has: the mean absolute in-plane Laplacian over the
    central slices, on lung and soft-tissue voxels (``lo`` < HU < ``hi``). Sharper kernels score
    higher. Only meaningful to compare two reconstructions of the SAME scan."""
    n = volume.shape[0]
    mid = volume[n // 5 : n - n // 5 or n].astype(np.float32)
    if mid.size == 0:
        mid = volume.astype(np.float32)
    lap = (
        4 * mid[:, 1:-1, 1:-1] - mid[:, :-2, 1:-1] - mid[:, 2:, 1:-1] - mid[:, 1:-1, :-2] - mid[:, 1:-1, 2:]
    )
    inner = mid[:, 1:-1, 1:-1]
    mask = (inner > lo) & (inner < hi)
    return float(np.abs(lap[mask]).mean()) if mask.any() else 0.0
