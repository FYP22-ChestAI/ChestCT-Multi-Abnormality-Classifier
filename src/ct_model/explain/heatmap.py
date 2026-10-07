"""L2 evidence: WHERE inside a slice the model looked for a label -- post hoc, on demand.

Grad-CAM (Selvaraju et al., ICCV 2017) applied to the ViT's last-block patch tokens:

    1. re-encode the slice's HU with the frozen stage-2 encoder, keeping the graph;
    2. put that embedding back into the volume's stored bag and run the trained classifier;
    3. back-propagate label c's logit to the last block's output tokens T (P patches x D);
    4. alpha = mean over patches of dlogit/dT;  cam_p = ReLU(sum_d alpha_d T_{p,d});
    5. reshape to the patch grid (14 x 14 for 224 / 16), upsample to the slice, scale to [0, 1].

Before any map is made, the re-encoded embedding must match the stored one (cosine >= ``min_cosine``):
otherwise the encoder, weights or volume is not what the classifier was trained on, and the map would
explain something else. The stored embeddings were computed in bfloat16 on the GPU, so they are close
to, not bit-identical with, a re-encode -- the measured cosine is always reported.

Two diagnostics (run them on the real model before trusting any maps; docs/model/README.md):
  * ``randomization_check`` (Adebayo et al., NeurIPS 2018, "Sanity Checks for Saliency Maps"): with the
    head's weights randomised the map must change -- a map that does not is not explaining the model.
  * ``deletion_check``: replacing the top-ranked patches (in model-input space) by the slice mean
    should lower the label's logit more than replacing the same number of random patches.

A map shows what the model used. It is NOT a validated segmentation of the finding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..models.volume_classifier import VolumeClassifier


class EvidenceMismatch(RuntimeError):
    """The re-encoded slice does not match the stored embedding the classifier saw."""


@dataclass
class SliceHeatmap:
    slice_index: int
    label: str
    heatmap: np.ndarray       # (H, W) in [0, 1], same size as the HU slice
    patch_cam: np.ndarray     # (gh, gw) before upsampling
    logit: float
    reencode_cosine: float


def _vit(encoder):
    model = getattr(encoder, "model", None)
    if model is None or not hasattr(model, "blocks") or not hasattr(model, "num_prefix_tokens"):
        raise NotImplementedError("in-slice heatmaps need a ViT encoder (ct_model.encoders.timm_vit)")
    return model


def _patch_size(vit) -> tuple[int, int]:
    p = vit.patch_embed.patch_size
    return (p, p) if isinstance(p, int) else tuple(p)


def _forward_with_tokens(encoder, x: torch.Tensor):
    """(embedding (1, D), last-block tokens (1, T, D) with .grad retained)."""
    vit = _vit(encoder)
    captured = {}

    def hook(_module, _inputs, output):
        output.retain_grad()
        captured["tokens"] = output

    handle = vit.blocks[-1].register_forward_hook(hook)
    try:
        emb = encoder.embed(x)
    finally:
        handle.remove()
    return emb, captured["tokens"]


def _logit_with_slice(model: VolumeClassifier, bag: torch.Tensor, k: int, emb: torch.Tensor, c: int) -> torch.Tensor:
    full = torch.cat([bag[:k], emb.to(bag.dtype), bag[k + 1:]])[None]
    mask = torch.ones(full.shape[:2], dtype=torch.bool, device=full.device)
    return model(full, mask).logits[0, c]


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def slice_heatmap(model: VolumeClassifier, encoder, bag: np.ndarray, hu_slice: np.ndarray, slice_index: int,
                  label: int | str, device: str | torch.device = "cpu", min_cosine: float = 0.99) -> SliceHeatmap:
    """``bag``: the volume's stored (n_slices, D) embeddings; ``hu_slice``: (H, W) HU of slice ``slice_index``."""
    names = model.label_names
    c = names.index(label) if isinstance(label, str) else int(label)
    model, encoder = model.eval(), encoder.eval()
    vit = _vit(encoder)
    stored = torch.as_tensor(np.asarray(bag, dtype=np.float32), device=device)
    with torch.enable_grad():
        x = encoder.input_transform(torch.as_tensor(np.asarray(hu_slice))[None].to(device))
        x.requires_grad_(True)
        emb, tokens = _forward_with_tokens(encoder, x)
        sim = cosine(emb.detach()[0], stored[slice_index])
        if sim < min_cosine:
            raise EvidenceMismatch(
                f"re-encoded slice {slice_index} has cosine {sim:.4f} with its stored embedding (< {min_cosine}): "
                "wrong encoder, weights or volume -- the map would not explain this model"
            )
        logit = _logit_with_slice(model, stored, slice_index, emb, c)
        logit.backward()
    model.zero_grad(set_to_none=True)  # explanation must leave no gradients behind
    n_prefix = vit.num_prefix_tokens
    t = tokens.detach()[0, n_prefix:]          # (P, D)
    g = tokens.grad.detach()[0, n_prefix:]      # (P, D)
    cam = F.relu((t * g.mean(dim=0)).sum(dim=-1))
    ph, pw = _patch_size(vit)
    gh, gw = x.shape[-2] // ph, x.shape[-1] // pw
    if gh * gw != cam.numel():
        raise RuntimeError(f"{cam.numel()} patch tokens do not form a {gh}x{gw} grid")
    grid = cam.reshape(1, 1, gh, gw)
    up = F.interpolate(grid, size=tuple(np.asarray(hu_slice).shape[-2:]), mode="bilinear", align_corners=False)[0, 0]
    peak = float(up.max())
    heat = (up / peak).cpu().numpy() if peak > 0 else np.zeros(up.shape, dtype=np.float32)
    return SliceHeatmap(slice_index, names[c], heat, grid[0, 0].cpu().numpy(), float(logit.detach()), sim)


def randomization_check(model: VolumeClassifier, encoder, bag, hu_slice, slice_index, label,
                        device="cpu", seed: int = 0) -> float:
    """Pearson correlation between the map of the trained head and of a head with random weights.
    Near 1 = the map does not depend on what the model learned (a failed sanity check)."""
    trained = slice_heatmap(model, encoder, bag, hu_slice, slice_index, label, device)
    original = {k: v.clone() for k, v in model.head.state_dict().items()}
    try:
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            w = model.head.linear.weight
            w.copy_(torch.randn(w.shape, generator=gen).to(w) * w.std())
        randomized = slice_heatmap(model, encoder, bag, hu_slice, slice_index, label, device)
    finally:
        model.head.load_state_dict(original)
    a, b = trained.patch_cam.ravel(), randomized.patch_cam.ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def deletion_check(model: VolumeClassifier, encoder, bag, hu_slice, slice_index, label, patch_cam: np.ndarray,
                   fraction: float = 0.1, n_random: int = 5, device="cpu", seed: int = 0) -> dict:
    """Logit drop when the top ``fraction`` of patches (by ``patch_cam``) are replaced by the slice's mean
    model-input value, against the mean drop for the same number of random patches."""
    names = model.label_names
    c = names.index(label) if isinstance(label, str) else int(label)
    model, encoder = model.eval(), encoder.eval()
    vit = _vit(encoder)
    stored = torch.as_tensor(np.asarray(bag, dtype=np.float32), device=device)
    x = encoder.input_transform(torch.as_tensor(np.asarray(hu_slice))[None].to(device))
    ph, pw = _patch_size(vit)
    gh, gw = patch_cam.shape
    n_del = max(1, int(round(fraction * gh * gw)))

    def logit_without(patches: np.ndarray) -> float:
        xm = x.clone()
        fill = x.mean()
        for p in patches:
            r, q = divmod(int(p), gw)
            xm[..., r * ph:(r + 1) * ph, q * pw:(q + 1) * pw] = fill
        return float(_logit_with_slice(model, stored, slice_index, encoder.embed(xm), c))

    base = float(_logit_with_slice(model, stored, slice_index, encoder.embed(x), c))
    top = np.argsort(-patch_cam.ravel(), kind="stable")[:n_del]
    rng = np.random.default_rng(seed)
    random_drops = [base - logit_without(rng.choice(gh * gw, n_del, replace=False)) for _ in range(n_random)]
    return {"logit": base, "drop_top": base - logit_without(top), "drop_random_mean": float(np.mean(random_drops)),
            "n_patches_deleted": n_del, "fraction": fraction}
