"""LoRA adaptation of the slice encoder (phase 2), for the sharp -> soft / NHRD domain gap.

LoRA (Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022, arXiv:2106.09685) keeps a
pretrained weight W frozen and learns a low-rank update:

    y = W x + (alpha / r) * B A x,   A: (r, in), B: (out, r), B = 0 at the start (so training starts at W)

Implemented here directly on ``nn.Linear`` (no peft dependency): ``apply_lora`` wraps the attention ``qkv``
and ``proj`` layers (``target_modules``; ``mlp.fc1`` / ``mlp.fc2`` also work) of the last ``last_n_blocks``
transformer blocks of a timm ViT, freezes every other parameter and leaves only A and B trainable.

An encoder config enables it with

    lora: {r: 8, alpha: 16, dropout: 0.05, target_modules: [qkv, proj], last_n_blocks: 12, adapter: <file>}

``adapter`` (a safetensors file written by ``save_adapter``) loads trained weights; without it, B = 0 and the
encoder is exactly the base model. The config, adapter path included, is part of the encoder fingerprint,
so embeddings of an adapted encoder go to their own store and never mix with the frozen ones; the adapter
file's sha256 is recorded in the store's provenance as well.

Training (ct_model.adapt) goes end to end from the HU cache with k sampled slices per volume; the frozen
blocks before the first adapted block run without gradients (``forward_features_split``), which is what
makes ViT-L trainable on a 16 GB GPU.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("qkv", "proj")
    last_n_blocks: int | None = None  # None = every block
    adapter: str | None = None        # trained weights (safetensors); None = untrained (B = 0)

    def __post_init__(self) -> None:
        self.target_modules = tuple(self.target_modules)
        if self.r < 1:
            raise ValueError(f"lora.r must be >= 1, got {self.r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"lora.dropout must be in [0, 1), got {self.dropout}")
        if self.last_n_blocks is not None and self.last_n_blocks < 1:
            raise ValueError(f"lora.last_n_blocks must be >= 1 or null, got {self.last_n_blocks}")


class LoRALinear(nn.Module):
    """``base(x) + scale * B(A(dropout(x)))`` with ``base`` frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # as nn.Linear (and the LoRA reference code)
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (self.dropout(x) @ self.lora_A.t().to(x.dtype)) @ self.lora_B.t().to(x.dtype)
        return self.base(x) + self.scale * update


def _blocks(model: nn.Module) -> nn.Sequential:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise TypeError(f"{type(model).__name__} has no .blocks -- LoRA here supports timm Vision Transformers")
    return blocks


def first_adapted_block(model: nn.Module, cfg: LoRAConfig) -> int:
    n = len(_blocks(model))
    return 0 if cfg.last_n_blocks is None else max(0, n - cfg.last_n_blocks)


def apply_lora(encoder, cfg: LoRAConfig):
    """Wrap the target Linear layers of the last ``last_n_blocks`` blocks of ``encoder.model`` with LoRA,
    load ``cfg.adapter`` if given, and leave only the LoRA parameters trainable. Returns the encoder."""
    model = encoder.model
    blocks = _blocks(model)
    start = first_adapted_block(model, cfg)
    wrapped = 0
    for p in encoder.parameters():
        p.requires_grad_(False)
    for i in range(start, len(blocks)):
        for name, module in list(blocks[i].named_modules()):
            leaf = name.rsplit(".", 1)[-1]
            if leaf in cfg.target_modules and isinstance(module, nn.Linear):
                parent = blocks[i].get_submodule(name.rsplit(".", 1)[0]) if "." in name else blocks[i]
                setattr(parent, leaf, LoRALinear(module, cfg.r, cfg.alpha, cfg.dropout))
                wrapped += 1
    if not wrapped:
        raise ValueError(f"no Linear layer named {list(cfg.target_modules)} in blocks {start}-{len(blocks) - 1}")
    encoder.lora_config = cfg
    encoder.lora_first_block = start
    if cfg.adapter:
        load_adapter(encoder, cfg.adapter)
    return encoder


def lora_parameters(module: nn.Module) -> dict[str, nn.Parameter]:
    return {n: p for n, p in module.named_parameters() if n.rsplit(".", 1)[-1] in ("lora_A", "lora_B")}


def save_adapter(encoder, path: str | Path, metadata: dict | None = None) -> Path:
    """Only the LoRA weights (a few MB), as safetensors; the LoRA config goes into the file's metadata."""
    import json

    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().float().cpu().contiguous() for n, p in lora_parameters(encoder).items()}
    cfg = asdict(encoder.lora_config) if hasattr(encoder, "lora_config") else {}
    cfg.pop("adapter", None)
    save_file(state, str(path), metadata={"lora": json.dumps(cfg), **{k: str(v) for k, v in (metadata or {}).items()}})
    return path


def load_adapter(encoder, path: str | Path) -> dict:
    """Load LoRA weights strictly: every LoRA parameter of the encoder must be in the file and vice versa."""
    from safetensors.torch import load_file

    state = load_file(str(path))
    params = lora_parameters(encoder)
    missing, unexpected = sorted(set(params) - set(state)), sorted(set(state) - set(params))
    if missing or unexpected:
        raise RuntimeError(f"adapter {path} does not match the encoder's LoRA layers: missing {missing[:5]}, "
                           f"unexpected {unexpected[:5]} -- check r / target_modules / last_n_blocks")
    with torch.no_grad():
        for n, p in params.items():
            if p.shape != state[n].shape:
                raise RuntimeError(f"adapter {path}: {n} has shape {tuple(state[n].shape)}, the encoder {tuple(p.shape)}")
            p.copy_(state[n].to(p.dtype))
    return {"n_tensors": len(state)}


def forward_features_split(model: nn.Module, x: torch.Tensor, first_trainable: int,
                           checkpoint: bool = True) -> torch.Tensor:
    """timm VisionTransformer.forward_features, with the blocks before ``first_trainable`` under no_grad
    (no activations kept) and the rest gradient-checkpointed. Same output as ``model.forward_features``."""
    from torch.utils.checkpoint import checkpoint as ckpt

    blocks = _blocks(model)
    with torch.no_grad():
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)
        for blk in blocks[:first_trainable]:
            x = blk(x)
    x = x.detach()
    for blk in blocks[first_trainable:]:
        x = ckpt(blk, x, use_reentrant=False) if checkpoint and torch.is_grad_enabled() else blk(x)
    return model.norm(x)
