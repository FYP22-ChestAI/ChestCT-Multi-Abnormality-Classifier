"""Shared GPU/CPU device resolution for spacing.py and resize.py.

Torch stays optional for this whole package -- if it isn't installed, or no
GPU is available, everything silently falls back to the CPU/scipy path that
has always worked. This is what lets the same preprocessing config file run
unchanged on a plain laptop and on a Colab GPU runtime (see docs/preprocessing/data_contract.md).
"""
from __future__ import annotations


def resolve_device(device: str) -> str:
    """Return the device to actually use: "cpu" or "cuda".

    device="cpu" always stays on CPU. device="cuda" or "auto" upgrades to
    "cuda" only if torch is installed AND a GPU is actually available;
    otherwise it quietly falls back to "cpu".
    """
    if device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if device in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"
