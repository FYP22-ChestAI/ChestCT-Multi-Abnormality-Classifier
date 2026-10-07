"""Any timm Vision Transformer as a SliceEncoder (DALE-CT-2S, DINOv2, ...), chosen by config.

Weights:
  * ``hf_safetensors`` -- a safetensors file in a Hugging Face repo, at a pinned ``revision``, loaded
    into ``timm.create_model(arch, **model_args)``. Loading is strict: every missing or unexpected
    key raises, except those under ``weights.ignore_prefixes``. (The DALE model card's own snippet
    uses ``strict=False``, which would silently leave a random layer in place if a key name differed.)
  * ``local_safetensors`` -- a safetensors file on disk (``weights.filename``), loaded the same strict way;
    its sha256 is recorded, since a local file has no commit.
  * ``timm_pretrained`` -- timm's own pretrained weights for ``arch``.
  * ``none`` -- random initialisation (tests, debugging).

Pooling (per slice, from ``forward_features``, i.e. after the final norm):
  * ``cls``        the class token -- identical to ``model(x)`` for ``global_pool="token"`` (the DALE card)
  * ``mean_patch`` mean of the patch tokens (prefix tokens -- class, registers -- excluded)
  * ``cls_mean``   both, concatenated (2x the dimension)
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..config import EncoderConfig, WeightsConfig
from .base import SliceEncoder
from .registry import register_encoder
from .transforms import build_input_transform


class TimmViTEncoder(SliceEncoder):
    def __init__(self, cfg: EncoderConfig, model, provenance: dict | None = None):
        dim = model.num_features * (2 if cfg.pooling == "cls_mean" else 1)
        super().__init__(cfg, build_input_transform(cfg.input), dim)
        self.model = model
        self._provenance = dict(provenance or {})
        self._check_input_compatible()

    def _check_input_compatible(self) -> None:
        patch = getattr(self.model.patch_embed, "patch_size", None)
        if patch is not None:
            ph, pw = (patch, patch) if isinstance(patch, int) else tuple(patch)
            h, w = self.cfg.input.size_hw
            if h % ph or w % pw:
                raise ValueError(
                    f"input.size_hw {self.cfg.input.size_hw} is not a multiple of {self.cfg.name}'s patch size {(ph, pw)}"
                )
        in_chans = self.model.patch_embed.proj.in_channels
        if in_chans != self.input_transform.channels:
            raise ValueError(
                f"{self.cfg.name} expects {in_chans} input channel(s) but input.transform "
                f"{self.cfg.input.transform} makes {self.input_transform.channels}"
            )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model.forward_features(x)  # (K, prefix + patches, D)
        if tokens.ndim != 3:
            raise RuntimeError(f"{self.cfg.arch}.forward_features returned shape {tuple(tokens.shape)}, expected (K, T, D)")
        cls = tokens[:, 0]
        if self.cfg.pooling == "cls":
            return cls
        patches = tokens[:, self.model.num_prefix_tokens:].mean(dim=1)
        return patches if self.cfg.pooling == "mean_patch" else torch.cat([cls, patches], dim=1)

    def provenance(self) -> dict:
        return dict(self._provenance)


def download_hf_weights(weights: WeightsConfig) -> tuple[Path, str | None]:
    """The local path of the weights file (downloaded once into the HF cache) and its resolved commit."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(weights.hf_repo, weights.filename, revision=weights.revision))
    commit = path.parent.name if path.parent.parent.name == "snapshots" else None
    return path, commit


def load_checkpoint(model: torch.nn.Module, path: str | Path, ignore_prefixes: list[str] = ()) -> dict:
    """Load a safetensors state dict strictly (see module docstring). Returns what was ignored."""
    from safetensors.torch import load_file

    state = load_file(str(path))
    missing, unexpected = model.load_state_dict(state, strict=False)  # shape mismatches still raise here

    def ignorable(key: str) -> bool:
        return any(key.startswith(p) for p in ignore_prefixes)

    bad_missing = [k for k in missing if not ignorable(k)]
    bad_unexpected = [k for k in unexpected if not ignorable(k)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"checkpoint {path} does not match the model: missing {bad_missing[:10]}"
            f"{' ...' if len(bad_missing) > 10 else ''}, unexpected {bad_unexpected[:10]}"
            f"{' ...' if len(bad_unexpected) > 10 else ''}. Fix arch/model_args, or list deliberate "
            "differences under weights.ignore_prefixes."
        )
    return {"ignored_missing": sorted(missing), "ignored_unexpected": sorted(unexpected), "n_tensors": len(state)}


@register_encoder("timm_vit")
def build_timm_vit(cfg: EncoderConfig) -> TimmViTEncoder:
    import timm

    weights = cfg.weights
    model = timm.create_model(cfg.arch, pretrained=weights.source == "timm_pretrained", **cfg.model_args)
    provenance: dict = {"timm_version": timm.__version__, "torch_version": torch.__version__}
    if weights.source == "hf_safetensors":
        path, commit = download_hf_weights(weights)
        report = load_checkpoint(model, path, weights.ignore_prefixes)
        provenance.update(weights_repo=weights.hf_repo, weights_file=weights.filename, weights_commit=commit,
                          weights_ignored=report["ignored_missing"] + report["ignored_unexpected"])
    elif weights.source == "local_safetensors":
        from ct_model.training.records import file_sha256

        report = load_checkpoint(model, weights.filename, weights.ignore_prefixes)
        provenance.update(weights_file=str(weights.filename), weights_sha256=file_sha256(weights.filename),
                          weights_ignored=report["ignored_missing"] + report["ignored_unexpected"])
    elif weights.source == "timm_pretrained":
        pcfg = getattr(model, "pretrained_cfg", {}) or {}
        provenance.update(weights_repo=pcfg.get("hf_hub_id"), weights_tag=pcfg.get("tag"))
    return TimmViTEncoder(cfg, model, provenance)
