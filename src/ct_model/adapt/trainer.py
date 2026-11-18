"""Phase 2: adapt the slice encoder with LoRA, end to end from the HU cache, with kernel consistency.

    HU slices (k per view) --encoder + LoRA--> slice embeddings --FeatureNorm--> aggregator --> head --> logits

There are no slice labels. The VOLUME loss trains the encoder: the chain is differentiable, so every sampled slice
gets its own gradient, weighted by the aggregator's attention on it. Two label-free terms target the domain gap:

    loss = BCE(every view)                                       the volume labels, as in phase 1
         + embedding_weight  * mean(1 - cos(e_sharp_k, e_soft_k))   aligned slices of the two views of a scan,
                                                                    on standardised (FeatureNorm) embeddings
         + prediction_weight * symmetric KL(p_sharp, p_soft)        the two views' label probabilities

A view is a real reconstruction (a sharp / soft pair of one scan) or, for a one-kernel scan, a kernel-simulating
augmentation of the same slices (ct_model.adapt.augment).

* Init: FeatureNorm, aggregator and head from a finished phase-1 run (``init_from``), so training starts from a
  working classifier and LoRA's B = 0 makes the encoder start exactly at the frozen one.
* Memory (16 GB): k slices per view (random_k), the blocks before the first adapted one without gradients, the
  adapted ones gradient-checkpointed, bf16 autocast, a few scans per pass and gradient accumulation.
* Selection: each epoch, val (uniform_k slices, both kernels) -> macro AUROC (``optim.monitor``) and the paired
  kernel metrics; the best epoch's adapter is kept.
* Output, outputs/adapters/<name>/<hash8>/seed<k>/:
    adapter.safetensors   the best epoch's LoRA weights (a few MB)
    encoder.yaml          the base encoder config + ``lora`` with this adapter: a new encoder for encode_volumes.py
    experiment.yaml       init_from's experiment with ``encoder: <encoder.yaml>`` for train_mil.py
    head.pt, metrics.csv, config.yaml, run_info.json, checkpoints/last.pt (resume: LoRA + classifier + optimiser, a few MB)

Then, with the adapted encoder frozen, the usual pipeline: encode its store, train the MIL head on full bags
(5 seeds), evaluate, clinical report -- so the result is compared with phase 1 on exactly the same yardstick.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from ct_preprocessing.config import load_config

from ..config import encoder_config_path, load_encoder_config
from ..data.slices import SliceSampler
from ..data.volumes import label_names, load_run_manifest, manifest_sha256, select_volumes
from ..encoders.lora import forward_features_split, lora_parameters, save_adapter
from ..encoders.registry import build_encoder
from ..models.volume_classifier import FeatureNorm, VolumeClassifier
from ..registry import LOSSES
from ..training import metrics as M
from ..training.records import append_row, ensure_dir, environment, git_info, is_finished, now_utc, read_json, write_json
from ..training.trainer import build_model, load_trained, lr_factor
from ..utils.device import resolve_device
from ..utils.log import say
from ..utils.seed import seed_everything
from .augment import kernel_augment
from .config import AdaptConfig
from .data import AdaptDataset, collate_items, scan_items

ADAPT_LEDGER = "index.csv"


class AlreadyAdapted(RuntimeError):
    """This config + seed has a finished adaptation run; results are never overwritten."""


@dataclass
class AdaptResult:
    dir: Path
    best_epoch: int
    best_value: float
    encoder_yaml: Path
    experiment_yaml: Path


def _amp(name: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or name == "float32":
        return None
    return {"bfloat16": torch.bfloat16, "float16": torch.float16}[name]


class EndToEnd(torch.nn.Module):
    """The adapted encoder in front of a phase-1 VolumeClassifier, for k-slice views."""

    def __init__(self, encoder, classifier: VolumeClassifier, checkpoint: bool = True):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.checkpoint = checkpoint

    def embed(self, hu: torch.Tensor) -> torch.Tensor:
        """(K, H, W) float HU -> (K, D) float32 slice embeddings."""
        x = self.encoder.input_transform(hu)
        tokens = forward_features_split(self.encoder.model, x, self.encoder.lora_first_block, self.checkpoint)
        return self.encoder.pool(tokens).float()

    def slice_cosine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-slice cosine of two views AFTER FeatureNorm -- the features the aggregator sees. Raw DALE
        embeddings share a large common component (raw sharp/soft cosine is ~0.96), which would hide the
        kernel difference; standardised, the discriminative dimensions dominate."""
        norm = self.classifier.feature_norm
        return F.cosine_similarity(norm(a), norm(b), dim=1)

    def classify(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """A list of (k_i, D) view embeddings -> (V, C) logits (padded bags, masked)."""
        k = max(e.shape[0] for e in embeddings)
        bags = embeddings[0].new_zeros((len(embeddings), k, embeddings[0].shape[1]))
        mask = torch.zeros((len(embeddings), k), dtype=torch.bool, device=bags.device)
        for i, e in enumerate(embeddings):
            bags[i, :e.shape[0]] = e
            mask[i, :e.shape[0]] = True
        return self.classifier(bags, mask).logits.float()


def sym_kl_bernoulli(logit_a: torch.Tensor, logit_b: torch.Tensor) -> torch.Tensor:
    """Mean over labels of KL(a||b) + KL(b||a) for Bernoulli distributions given by logits."""
    la, lb = F.logsigmoid(logit_a), F.logsigmoid(logit_b)
    na, nb = F.logsigmoid(-logit_a), F.logsigmoid(-logit_b)
    pa, pb = la.exp(), lb.exp()
    kl_ab = pa * (la - lb) + (1 - pa) * (na - nb)
    kl_ba = pb * (lb - la) + (1 - pb) * (nb - na)
    return (kl_ab + kl_ba).mean()


def _views_on_device(item: dict, device: torch.device, cfg: AdaptConfig, gen: torch.Generator | None) -> list[torch.Tensor]:
    views = [torch.from_numpy(v["hu"]).to(device, non_blocking=True).float() for v in item["views"]]
    if item["augment"]:
        c = cfg.consistency
        views.append(kernel_augment(views[0], item["augment"], gen, c.blur_sigma, c.sharpen_amount))
    return views


def batch_loss(model: EndToEnd, batch: list[dict], loss_fn, cfg: AdaptConfig, device: torch.device,
               amp_dtype: torch.dtype | None, gen: torch.Generator | None) -> tuple[torch.Tensor, dict]:
    hus, owners = [], []
    for j, item in enumerate(batch):
        for v in _views_on_device(item, device, cfg, gen):
            hus.append(v)
            owners.append(j)
    sizes = [h.shape[0] for h in hus]
    with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        emb = model.embed(torch.cat(hus))
    embs = list(torch.split(emb, sizes))
    logits = model.classify(embs)
    labels = torch.stack([torch.from_numpy(batch[j]["labels"]) for j in owners]).to(device)
    known = ~torch.isnan(labels)
    bce = loss_fn(logits, torch.nan_to_num(labels), known)

    emb_terms, pred_terms = [], []
    for j, item in enumerate(batch):
        rows = [i for i, o in enumerate(owners) if o == j]
        if len(rows) != 2:
            continue
        a, b = rows
        pred_terms.append(sym_kl_bernoulli(logits[a], logits[b]))
        if item["aligned"] and embs[a].shape == embs[b].shape:
            emb_terms.append((1 - model.slice_cosine(embs[a], embs[b])).mean())
    zero = logits.new_zeros(())
    emb_loss = torch.stack(emb_terms).mean() if emb_terms else zero
    pred_loss = torch.stack(pred_terms).mean() if pred_terms else zero
    c = cfg.consistency
    loss = bce + c.embedding_weight * emb_loss + c.prediction_weight * pred_loss
    return loss, {"bce": float(bce.detach()), "emb": float(emb_loss.detach()), "pred": float(pred_loss.detach()),
                  "views": len(hus), "slices": int(sum(sizes))}


@torch.no_grad()
def validate(model: EndToEnd, dataset: AdaptDataset, loss_fn, cfg: AdaptConfig, device: torch.device,
             amp_dtype: torch.dtype | None, workers: int = 4) -> dict:
    """Per VOLUME (each real view): probabilities -> macro AUROC (overall, per kernel), loss; per pair: the
    cosine of aligned slices, |dp|, and how often a decision at 0.5 changes."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=workers, collate_fn=collate_items)
    probs, ys, kernels, losses, cos, dps, flips = [], [], [], [], [], [], []
    for (item,) in loader:
        hus = [torch.from_numpy(v["hu"]).to(device).float() for v in item["views"]]
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            embs = [model.embed(h) for h in hus]
        logits = model.classify(embs)
        y = torch.from_numpy(item["labels"]).to(device).expand_as(logits)
        known = ~torch.isnan(y)
        losses.append(float(loss_fn(logits, torch.nan_to_num(y), known)))
        p = torch.sigmoid(logits).cpu().numpy()
        for i, v in enumerate(item["views"]):
            probs.append(p[i]); ys.append(item["labels"]); kernels.append(v["kernel"])
        if len(hus) == 2:
            dps.append(np.abs(p[0] - p[1]).mean())
            flips.append(((p[0] >= 0.5) != (p[1] >= 0.5)).mean())
            if item["aligned"] and embs[0].shape == embs[1].shape:
                cos.append(float(model.slice_cosine(embs[0], embs[1]).mean()))
    model.train()
    P, Y, K = np.stack(probs), np.stack(ys).astype(np.float64), np.asarray(kernels)
    out = {"val_macro_auroc": M.macro_auroc(Y, P), "val_loss": float(np.mean(losses)), "val_volumes": len(P),
           "val_pairs": len(dps), "val_pair_cosine": float(np.mean(cos)) if cos else np.nan,
           "val_pair_abs_dp": float(np.mean(dps)) if dps else np.nan,
           "val_pair_flip_rate": float(np.mean(flips)) if flips else np.nan}
    for kc in ("sharp", "soft"):
        if (K == kc).any():  # NaN when a kernel has too few volumes to score
            out[f"val_macro_auroc[{kc}]"] = M.macro_auroc(Y[K == kc], P[K == kc])
    return out


def trainable_state(model: EndToEnd) -> dict:
    """What changes during adaptation: the LoRA matrices and the classifier (FeatureNorm buffers included).
    The frozen ViT weights (1.2 GB) come from the base checkpoint, so the resume state stays a few MB."""
    lora = set(lora_parameters(model.encoder))
    return {k: v for k, v in model.state_dict().items()
            if k.startswith("classifier.") or k.removeprefix("encoder.") in lora}


def load_trainable_state(model: EndToEnd, state: dict) -> None:
    expected = set(trainable_state(model))
    missing, unexpected = expected - set(state), set(state) - set(model.state_dict())
    if missing or unexpected:
        raise RuntimeError(f"resume state does not match the model: missing {sorted(missing)[:5]}, unexpected {sorted(unexpected)[:5]}")
    model.load_state_dict(state, strict=False)


def _fit_norm(model: EndToEnd, dataset: AdaptDataset, n: int, device, amp_dtype, log) -> None:
    """Without init_from: FeatureNorm from the starting encoder's embeddings of ``n`` train scans."""
    bags = []
    with torch.no_grad():
        for i in range(min(n, len(dataset))):
            hu = torch.from_numpy(dataset[i]["views"][0]["hu"]).to(device).float()
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                bags.append(model.embed(hu).cpu().numpy())
    model.classifier.feature_norm.fit(bags)
    log(f"FeatureNorm fitted on {len(bags)} train volumes (no init_from)")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_outputs(cfg: AdaptConfig, out: Path, enc_cfg, init_cfg: dict | None) -> tuple[Path, Path]:
    """encoder.yaml (base config + lora with the adapter) and experiment.yaml (for train_mil.py)."""
    base = yaml.safe_load(encoder_config_path(cfg.encoder).read_text(encoding="utf-8")) or {}
    base["name"] = f"{enc_cfg.name}_{cfg.name}"
    base["lora"] = {**cfg.lora, "adapter": _relative(out / "adapter.safetensors")}
    enc_path = out / "encoder.yaml"
    enc_path.write_text(
        f"# {cfg.encoder} adapted with LoRA by scripts/model/adapt_encoder.py ({_relative(out)}).\n"
        "# Encode its store with:  python scripts/model/encode_volumes.py --encoder " + _relative(enc_path)
        + f" --run {cfg.data.run}\n\n" + yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    exp = dict(init_cfg or {})
    exp.update(name=f"{(init_cfg or {}).get('name', 'abmil')}_{cfg.name}", encoder=_relative(enc_path), seed=0)
    exp.setdefault("data_config", cfg.data_config)
    exp.setdefault("encode_config", cfg.encode_config)
    exp.setdefault("aggregator", cfg.aggregator)
    exp.setdefault("head", cfg.head)
    data = dict(exp.get("data") or {})
    data.update(source=cfg.data.source, run=cfg.data.run, train_kernel_classes=list(cfg.data.kernel_classes) if cfg.data.kernel_classes else None)
    exp["data"] = data
    exp.pop("device", None)
    exp_path = out / "experiment.yaml"
    exp_path.write_text(
        f"# Phase-1 MIL head on the LoRA-adapted encoder of {_relative(out)} (encoder frozen again).\n"
        "#   python scripts/model/train_mil.py --experiment " + _relative(exp_path) + " --seed 0\n\n"
        + yaml.safe_dump(exp, sort_keys=False), encoding="utf-8")
    return enc_path, exp_path


def train_adapter(cfg: AdaptConfig, log: Callable[[str], None] = say, workers: int = 4) -> AdaptResult:
    data_cfg = load_config(cfg.data_config)
    run, manifest = load_run_manifest(data_cfg, cfg.data.source, cfg.data.run)
    cfg_hash = cfg.config_hash(manifest_sha256(run))
    out = cfg.run_dir(cfg_hash)
    rid = f"{cfg.name}/{cfg_hash[:8]}/seed{cfg.seed}"
    if is_finished(out):
        raise AlreadyAdapted(f"{rid} is already adapted ({out}); use another --seed or change a setting")
    ensure_dir(out / "checkpoints")
    last_path = out / "checkpoints" / "last.pt"
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    amp_dtype = _amp(cfg.amp_dtype, device)

    enc_cfg = load_encoder_config(cfg.encoder)
    encoder = build_encoder(replace(enc_cfg, lora=dict(cfg.lora)), frozen=False)
    init_cfg = None
    if cfg.init_from:
        trained = load_trained(cfg.init_from, device="cpu")
        classifier, names = trained.model, trained.label_names
        init_cfg = trained.cfg.to_dict()
        if trained.cfg.encoder != cfg.encoder and Path(trained.cfg.encoder).stem != cfg.encoder:
            log(f"note: init_from was trained on encoder {trained.cfg.encoder}, adapting {cfg.encoder}")
        in_dim = classifier.feature_norm.mean.numel()
        if in_dim != encoder.embed_dim:
            raise ValueError(f"init_from expects {in_dim}-d slice features but {cfg.encoder} makes {encoder.embed_dim}-d")
    else:
        names = list(cfg.data.labels or label_names(manifest))
        from ..training.config import parse_experiment
        stub = parse_experiment({"name": "stub", "aggregator": cfg.aggregator, "head": cfg.head})
        classifier = build_model(stub, encoder.embed_dim, names)
    available = label_names(manifest)
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"labels {missing} are not in the manifest of run {run.name}")
    label_index = [available.index(n) for n in names]

    def records(split):
        return select_volumes(manifest, data_cfg.paths.cache_dir, splits=(split,), qc_passed_only=cfg.data.qc_passed_only,
                              kernel_classes=cfg.data.kernel_classes)

    train_items = scan_items(records("train"), manifest, cfg.consistency.pairs)
    val_items = scan_items(records("val"), manifest, True)
    s = cfg.slices
    train_ds = AdaptDataset(train_items, label_index, SliceSampler("random_k", k=s.train_k), max_offset=s.max_offset,
                            augment_prob=cfg.consistency.augment_prob, seed=cfg.seed)
    val_ds = AdaptDataset(val_items, label_index, SliceSampler("uniform_k", k=s.val_k), max_offset=s.max_offset, seed=cfg.seed)
    n_pairs = sum(len(i.views) == 2 for i in train_items)
    model = EndToEnd(encoder, classifier, cfg.optim.checkpoint).to(device)
    if not cfg.init_from:
        _fit_norm(model, train_ds, cfg.norm_fit_volumes, device, amp_dtype, log)
    model.train()

    o = cfg.optim
    lora = list(lora_parameters(encoder).values())
    head = [p for p in classifier.parameters() if p.requires_grad]
    groups = [{"params": lora, "lr": o.lr_lora, "weight_decay": o.weight_decay}]
    groups += [{"params": [p for p in head if p.ndim >= 2], "lr": o.lr_head, "weight_decay": o.weight_decay},
               {"params": [p for p in head if p.ndim < 2], "lr": o.lr_head, "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups)
    n_epoch = min(o.scans_per_epoch or len(train_ds), len(train_ds))
    passes = math.ceil(n_epoch / o.scans_per_step)
    steps_per_epoch = math.ceil(passes / o.accumulation)
    total = steps_per_epoch * o.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda st: lr_factor(st, o.warmup_steps, total))
    loss_fn = LOSSES.build({"type": "bce"}, train_labels=None).to(device)

    start_epoch, best_value, best_epoch, history = 0, (math.inf if o.monitor == "val_loss" else -math.inf), 0, []
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=True)
        if "model" in state:  # older runs saved the whole model, frozen ViT included
            model.load_state_dict(state["model"])
        else:
            load_trainable_state(model, state["trainable"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_value, best_epoch, history = state["epoch"], state["best_value"], state["best_epoch"], state["history"]
        log(f"resuming {rid} after epoch {start_epoch}")
    else:
        write_json(out / "run_info.json", {
            "run_id": rid, "config_hash": cfg_hash, "status": "running", "started": now_utc(), "git": git_info(),
            "environment": environment(),
            "data": {"source": run.source, "run": run.name, "manifest": str(run.manifest_path),
                     "train_scans": len(train_items), "train_pairs": n_pairs, "val_scans": len(val_items),
                     "val_pairs": sum(len(i.views) == 2 for i in val_items)},
            "lora_parameters": int(sum(p.numel() for p in lora)), "label_names": names,
        })
        (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    log(f"{rid}: {len(train_items)} train scans ({n_pairs} sharp/soft pairs), {len(val_items)} val scans; "
        f"LoRA {sum(p.numel() for p in lora) / 1e6:.2f} M parameters in blocks {encoder.lora_first_block}+; "
        f"{steps_per_epoch} optimiser steps per epoch x {o.epochs} epochs; device {device}, autocast {amp_dtype}")

    if start_epoch == 0 and not history:  # the starting point: LoRA B = 0, i.e. the frozen encoder
        v = validate(model, val_ds, loss_fn, cfg, device, amp_dtype, workers)
        history.append({"epoch": 0, **v})
        log(f"  epoch   0 (frozen start)  val macro AUROC {v['val_macro_auroc']:.4f}  pair cosine {v['val_pair_cosine']:.4f}  "
            f"pair flips@0.5 {v['val_pair_flip_rate']:.4f}")
    gen = torch.Generator().manual_seed(cfg.seed * 1_000_003 + start_epoch)  # augmentation strengths (CPU draws)
    for epoch in range(start_epoch, o.epochs):
        t0 = time.perf_counter()
        train_ds.epoch = epoch
        order = np.random.default_rng((cfg.seed, epoch)).permutation(len(train_ds))[:n_epoch]
        loader = DataLoader(Subset(train_ds, order.tolist()), batch_size=o.scans_per_step, shuffle=False,
                            num_workers=workers, collate_fn=collate_items, prefetch_factor=4 if workers else None,
                            persistent_workers=False)
        sums, slices, passes_done = {"bce": 0.0, "emb": 0.0, "pred": 0.0}, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for b, batch in enumerate(loader):
            loss, parts = batch_loss(model, batch, loss_fn, cfg, device, amp_dtype, gen)
            (loss / o.accumulation).backward()
            for k in sums:
                sums[k] += parts[k]
            slices += parts["slices"]
            passes_done += 1
            if passes_done % o.accumulation == 0 or b == len(loader) - 1:
                if o.grad_clip:
                    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], o.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if passes_done % 200 == 0:
                rate = slices / (time.perf_counter() - t0)
                left = (len(loader) - passes_done) * slices / passes_done / max(rate, 1e-9) / 60
                log(f"    epoch {epoch + 1} pass {passes_done}/{len(loader)}  bce {sums['bce'] / passes_done:.4f}  "
                    f"emb {sums['emb'] / passes_done:.4f}  pred {sums['pred'] / passes_done:.4f}  {rate:.0f} slices/s  "
                    f"~{left:.0f} min left")
        v = validate(model, val_ds, loss_fn, cfg, device, amp_dtype, workers)
        value = v[o.monitor]
        improved = value < best_value if o.monitor == "val_loss" else value > best_value
        if improved:
            best_value, best_epoch = float(value), epoch + 1
            save_adapter(encoder, out / "adapter.safetensors", {"epoch": epoch + 1, "run_id": rid})
            torch.save({"classifier": classifier.state_dict(), "label_names": names, "epoch": epoch + 1}, out / "head.pt")
        history.append({"epoch": epoch + 1, "lr_lora": scheduler.get_last_lr()[0],
                        **{f"train_{k}": val / max(passes_done, 1) for k, val in sums.items()}, **v,
                        "improved": improved, "seconds": round(time.perf_counter() - t0, 1)})
        pd.DataFrame(history).to_csv(out / "metrics.csv", index=False, lineterminator="\n")
        torch.save({"trainable": trainable_state(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1, "best_value": best_value, "best_epoch": best_epoch, "history": history}, last_path)
        log(f"  epoch {epoch + 1:3d}  bce {sums['bce'] / max(passes_done, 1):.4f}  val macro AUROC {v['val_macro_auroc']:.4f}  "
            f"pair cosine {v['val_pair_cosine']:.4f}  pair flips@0.5 {v['val_pair_flip_rate']:.4f}"
            f"{'  *best' if improved else ''}  ({(time.perf_counter() - t0) / 60:.1f} min)")

    if not (out / "adapter.safetensors").is_file():
        raise RuntimeError("no epoch improved on the starting point -- nothing to save (check the learning rates)")
    enc_path, exp_path = _write_outputs(cfg, out, enc_cfg, init_cfg)
    info = read_json(out / "run_info.json")
    info.update(status="finished", ended=now_utc(), best_epoch=best_epoch, monitor=o.monitor, best_value=best_value,
                epochs_run=len(history) - 1, encoder_yaml=_relative(enc_path), experiment_yaml=_relative(exp_path))
    write_json(out / "run_info.json", info)
    append_row(Path(cfg.output_dir) / ADAPT_LEDGER, {
        "run_id": rid, "name": cfg.name, "config_hash": cfg_hash, "seed": cfg.seed, "finished": info["ended"],
        "data_run": run.name, "encoder": cfg.encoder, "init_from": cfg.init_from, "best_epoch": best_epoch,
        "monitor": o.monitor, "best_value": best_value, "start_value": history[0].get(o.monitor),
        "dir": out.as_posix(),
    })
    log(f"finished {rid}: best epoch {best_epoch} ({o.monitor} {best_value:.4f}); adapter {out / 'adapter.safetensors'}")
    log("next, with the adapted encoder frozen:")
    log(f"  python scripts/model/encode_volumes.py --encoder {_relative(enc_path)} --run {run.name}")
    log(f"  python scripts/model/train_mil.py --experiment {_relative(exp_path)} --seed 0   (and seeds 1-4)")
    return AdaptResult(out, best_epoch, best_value, enc_path, exp_path)
