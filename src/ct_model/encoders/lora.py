"""TODO (phase 2): LoRA adaptation of the slice encoder, for the sharp -> soft / NHRD domain gap.

Intended design (for whoever implements it):

* ``apply_lora(encoder, cfg)`` wraps the backbone's attention ``qkv`` and ``proj`` Linear layers (timm
  ViT block names: ``blocks.{i}.attn.qkv`` / ``blocks.{i}.attn.proj``; optionally ``mlp.fc1`` / ``mlp.fc2``)
  of the last ``last_n_blocks`` blocks with peft's LoRA (``peft.LoraConfig`` + ``inject_adapter_in_model``),
  freezes everything else, and returns the encoder with only the adapter weights trainable.
* An encoder config enables it with ``lora: {r: 8, alpha: 16, ...}``. That changes the encoder's
  fingerprint, so frozen phase-1 embeddings are never confused with adapted ones.
* Training goes end to end from the HU cache (ct_model.data.datasets.HUVolumeDataset), NOT from the
  embedding store. Memory on a 16 GB GPU: sample k slices per volume (SliceSampler random_k, k ~ 32-64),
  bf16 autocast, ``model.set_grad_checkpointing(True)``, batch of 1-2 volumes + gradient accumulation.
* Save only the adapter weights (a few MB) with the experiment, plus the base weights commit.
* Initialise stages 3 + 4 from the phase-1 checkpoint, then train adapters (+ optionally the head).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("qkv", "proj")
    last_n_blocks: int | None = None  # None = every block


def apply_lora(encoder, cfg: LoRAConfig):
    raise NotImplementedError(
        "LoRA is phase 2 and not implemented yet -- set `lora: null` in the encoder config. "
        "See the design notes in ct_model/encoders/lora.py and docs/model/architecture.md."
    )
