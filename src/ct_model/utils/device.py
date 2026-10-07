"""Device and mixed-precision choices, made once and printed, so a run says what it really used."""
from __future__ import annotations

import torch

_AMP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": None}


def resolve_device(device: str) -> torch.device:
    """``auto`` -> cuda when a GPU is present, else cpu. ``cuda`` without a GPU is an error, not a
    silent fallback: a 10k-volume encode on the CPU would take days."""
    if device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if device == "cuda":
        raise RuntimeError("device cuda was asked for but torch sees no GPU (check the driver / CUDA build of torch)")
    return torch.device("cpu")


def resolve_amp_dtype(amp_dtype: str, device: torch.device) -> torch.dtype | None:
    """The autocast dtype, or None for plain float32. ``auto`` = bfloat16 on a GPU that supports it
    (Ampere and newer, e.g. the RTX 4070 on the server), else float32 (CPU autocast is slow)."""
    if amp_dtype == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    return _AMP[amp_dtype]


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    props = torch.cuda.get_device_properties(device)
    return f"cuda ({props.name}, {props.total_memory / 2**30:.1f} GiB)"
