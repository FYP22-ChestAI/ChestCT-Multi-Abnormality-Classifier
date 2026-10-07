"""HU -> backbone input. Each backbone declares its own transform in its encoder config, because
backbones disagree: DALE-CT wants one z-scored channel, ImageNet-pretrained models want the data
contract's three HU windows as RGB plus ImageNet normalisation.

Everything here is a torch module that runs on whatever device the slices are on, so int16 slices
travel to the GPU and are converted there. Input: (K, H, W) HU (any dtype). Output: (K, C, h, w) float32.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import InputConfig


class ClipZScore(nn.Module):
    """``(clip(HU, low, high) - mean) / std`` -- one channel.

    This is exactly DALE-CT's ``CTInferenceTransform`` (clip, map to [0, 1], z-score with the [0, 1]-scaled
    mean and std): the [0, 1] mapping cancels out, ((x-lo)/R - (m-lo)/R) / (s/R) = (x-m)/s.
    """

    def __init__(self, clip_hu: tuple[float, float], mean_hu: float, std_hu: float):
        super().__init__()
        self.low, self.high = float(clip_hu[0]), float(clip_hu[1])
        self.mean, self.std = float(mean_hu), float(std_hu)

    def forward(self, hu: torch.Tensor) -> torch.Tensor:
        x = hu.float().clamp_(self.low, self.high)
        return ((x - self.mean) / self.std).unsqueeze(1)


class MultiWindow(nn.Module):
    """One channel per HU window, each scaled to [0, 1], then optional per-channel (x - mean) / std."""

    def __init__(self, windows: list[tuple[float, float]], channel_mean=None, channel_std=None):
        super().__init__()
        self.windows = [(float(lo), float(hi)) for lo, hi in windows]
        c = len(self.windows)
        mean = torch.tensor(channel_mean if channel_mean is not None else [0.0] * c).view(1, c, 1, 1)
        std = torch.tensor(channel_std if channel_std is not None else [1.0] * c).view(1, c, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, hu: torch.Tensor) -> torch.Tensor:
        x = hu.float()
        chans = [(x.clamp(lo, hi) - lo) / (hi - lo) for lo, hi in self.windows]
        return (torch.stack(chans, dim=1) - self.mean.to(x.device)) / self.std.to(x.device)


class InputTransform(nn.Module):
    """Intensity transform, then resize to ``size_hw`` if the slices are not that size already."""

    def __init__(self, cfg: InputConfig):
        super().__init__()
        if cfg.transform == "clip_zscore":
            self.intensity = ClipZScore(cfg.clip_hu, cfg.mean_hu, cfg.std_hu)
        else:
            self.intensity = MultiWindow(cfg.windows, cfg.channel_mean, cfg.channel_std)
        self.size_hw = tuple(cfg.size_hw)
        self.resize_mode = cfg.resize_mode
        self.channels = cfg.channels

    def forward(self, hu: torch.Tensor) -> torch.Tensor:
        if hu.ndim != 3:
            raise ValueError(f"expected (K, H, W) HU slices, got shape {tuple(hu.shape)}")
        x = self.intensity(hu)
        if tuple(x.shape[-2:]) != self.size_hw:
            kwargs = {} if self.resize_mode == "nearest" else {"align_corners": False, "antialias": True}
            x = F.interpolate(x, size=self.size_hw, mode=self.resize_mode, **kwargs)
        return x


def build_input_transform(cfg: InputConfig) -> InputTransform:
    return InputTransform(cfg)
