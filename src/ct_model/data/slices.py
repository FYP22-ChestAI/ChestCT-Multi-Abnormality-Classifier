"""Which slices of a volume a model sees.

Phase 1 (frozen encoder, precomputed embeddings) uses ``all``: stage 2 encodes every slice once and
stage 3 can still subsample the bag. The other modes exist for phase 2 (LoRA), where gradients flow
through the encoder and a 16 GB GPU cannot hold ViT-L activations for ~230 slices per volume:

    all        every slice
    stride     every ``stride``-th slice (1.5 mm cache spacing -> stride 2 = 3 mm)
    uniform_k  ``k`` slices evenly spread head to foot (deterministic: validation / test)
    random_k   ``k`` slices, one drawn at random from each of ``k`` equal segments (training: keeps
               head-to-foot coverage while varying the slices every epoch)

Indices are always returned sorted, so slice order (head -> foot) is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MODES = ("all", "stride", "uniform_k", "random_k")


@dataclass(frozen=True)
class SliceSampler:
    mode: str = "all"
    stride: int = 1
    k: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"slice sampler mode must be one of {list(_MODES)}, got {self.mode!r}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if self.mode in ("uniform_k", "random_k") and (self.k is None or self.k < 1):
            raise ValueError(f"mode {self.mode} needs k >= 1, got {self.k}")

    def __call__(self, n_slices: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if n_slices < 1:
            raise ValueError(f"a volume needs at least one slice, got {n_slices}")
        if self.mode == "all":
            return np.arange(n_slices)
        if self.mode == "stride":
            return np.arange(0, n_slices, self.stride)
        if n_slices <= self.k:
            return np.arange(n_slices)
        edges = np.linspace(0, n_slices, self.k + 1)
        if self.mode == "uniform_k":
            return np.floor((edges[:-1] + edges[1:]) / 2).astype(np.int64)
        rng = rng if rng is not None else np.random.default_rng()
        lo, hi = np.ceil(edges[:-1]).astype(np.int64), np.ceil(edges[1:]).astype(np.int64)
        return lo + (rng.random(self.k) * (hi - lo)).astype(np.int64)
