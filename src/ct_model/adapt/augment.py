"""Kernel-simulating augmentation, on HU slices on the GPU (before the encoder's own input transform).

A soft (mediastinal) kernel is, to first order, a low-pass version of a sharp (lung) kernel of the same
acquisition; a sharp kernel boosts high frequencies. So:

    blur(x, sigma)              Gaussian blur, sigma in pixels      sharp -> "soft"
    sharpen(x, amount, sigma)   x + amount * (x - blur(x, sigma))   soft  -> "sharp" (unsharp mask)

Crude next to the real kernel pairs, but it covers the scans that have only one reconstruction.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, device=None, dtype=torch.float32) -> torch.Tensor:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """(K, H, W) -> (K, H, W), separable Gaussian, reflect padding."""
    k = gaussian_kernel1d(sigma, x.device, x.dtype)
    r = (k.numel() - 1) // 2
    y = x.unsqueeze(1)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    return y.squeeze(1)


def sharpen(x: torch.Tensor, amount: float, sigma: float = 1.0) -> torch.Tensor:
    return x + amount * (x - blur(x, sigma))


def kernel_augment(hu: torch.Tensor, kind: str, generator: torch.Generator | None,
                   blur_sigma: tuple[float, float], sharpen_amount: tuple[float, float]) -> torch.Tensor:
    """Float HU (K, H, W) -> the simulated other kernel. ``kind`` is "blur" or "sharpen"."""
    u = float(torch.rand((), generator=generator))
    if kind == "blur":
        return blur(hu, blur_sigma[0] + u * (blur_sigma[1] - blur_sigma[0]))
    if kind == "sharpen":
        return sharpen(hu, sharpen_amount[0] + u * (sharpen_amount[1] - sharpen_amount[0]))
    raise ValueError(f"unknown kernel augmentation {kind!r}")
