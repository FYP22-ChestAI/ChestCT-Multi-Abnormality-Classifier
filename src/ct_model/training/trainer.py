"""Phase 1 training: aggregator + head on frozen slice embeddings. Train and VAL only -- the test
split is never read here (ct_model.training.evaluate does that, once, for a finished run).

    fit FeatureNorm on train slices; head bias <- train prevalence; loss built with train labels
    for each epoch (seeded shuffle):  AdamW step per batch, LR warm-up then cosine, grad clipping
        val: loss, macro / micro AUROC, macro AUPRC, per-label AUROC  -> metrics.csv
        best on `optim.monitor` -> checkpoints/best.pt;  every epoch -> checkpoints/last.pt (resume)
        stop after `optim.patience` epochs without improvement
    best model on val -> predictions_val.csv, max-F1 thresholds (thresholds.json), val metrics
    finished -> one row in outputs/experiments/index.csv

Running the same command again resumes an unfinished run from last.pt; a finished run is refused
(a new seed, or any changed setting -- a new config hash -- gives a new folder).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from ..data.datasets import EmbeddingBagDataset, collate_bags
from ..data.slices import SliceSampler
from ..models.volume_classifier import FeatureNorm, VolumeClassifier
from ..registry import AGGREGATORS, HEADS, LOSSES
from ..utils.device import resolve_device
from ..utils.log import say
from ..utils.seed import seed_everything
from . import metrics as M
from .config import ExperimentConfig, parse_experiment
from .data import ExperimentData, open_data
from .records import (
    TRAIN_LEDGER, append_row, config_hash, ensure_dir, environment, git_info, is_finished, now_utc, read_json,
    run_dir, run_id, write_json,
)


class AlreadyTrained(RuntimeError):
    """This config + seed has a finished run; results are never overwritten."""


def build_model(cfg: ExperimentConfig, in_dim: int, label_names: list[str]) -> VolumeClassifier:
    n = len(label_names)
    aggregator = AGGREGATORS.build(cfg.aggregator, in_dim=in_dim, n_labels=n)
    head = HEADS.build(cfg.head, in_dim=aggregator.out_dim, n_labels=n)
    if aggregator.per_label and head.n_labels != n:
        raise ValueError("a per-label aggregator needs one head output per label")
    return VolumeClassifier(FeatureNorm(in_dim), aggregator, head, label_names)


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on weight matrices only -- not on biases or 1-D parameters."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warm-up to 1 over ``warmup_steps``, then cosine decay to 0 at ``total_steps``."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def predict(model: VolumeClassifier, dataset: EmbeddingBagDataset, device: torch.device, batch_size: int = 32,
            loss_fn: Callable | None = None) -> tuple[np.ndarray, float | None]:
    """(probabilities (n_volumes, C) in dataset order, mean loss or None)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_bags, num_workers=0)
    probs, total, count = [], 0.0, 0
    for batch in loader:
        logits = model(batch["bags"].to(device), batch["mask"].to(device)).logits
        if loss_fn is not None:
            n = int(batch["label_mask"].sum())
            total += float(loss_fn(logits, batch["labels"].to(device), batch["label_mask"].to(device))) * n
            count += n
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs), (total / count if loss_fn is not None and count else None)


def predictions_frame(dataset: EmbeddingBagDataset, probs: np.ndarray, label_names: list[str]) -> pd.DataFrame:
    y = dataset.labels()
    frame = pd.DataFrame({
        "volume_id": [r.volume_id for r in dataset.records],
        "patient_id": [r.patient_id for r in dataset.records],
        "split": [r.split for r in dataset.records],
        "kernel_class": [r.kernel_class for r in dataset.records],
    })
    for c, name in enumerate(label_names):
        frame[f"prob_{name}"] = probs[:, c]
        frame[f"true_{name}"] = y[:, c]
    return frame


@dataclass
class TrainResult:
    run_id: str
    dir: Path
    best_epoch: int
    best_value: float
    epochs_run: int
    stopped: str
    val_summary: dict


def _better(value: float, best: float, monitor: str) -> bool:
    if np.isnan(value):
        return False
    return value < best if monitor == "val_loss" else value > best


def train_experiment(cfg: ExperimentConfig, log: Callable[[str], None] = say) -> TrainResult:
    data: ExperimentData = open_data(cfg)
    kernels = cfg.data.train_kernel_classes
    sampler = SliceSampler(**cfg.data.train_slice_sampler)
    train_ds = data.dataset("train", kernels, preload=cfg.data.preload, sampler=sampler, seed=cfg.seed)
    val_ds = data.dataset("val", kernels, preload=cfg.data.preload)
    names = data.label_names

    cfg_hash = config_hash(cfg, data.store.dir.name, data.manifest_sha256)
    out = run_dir(cfg, cfg_hash)
    rid = run_id(cfg, cfg_hash)
    if is_finished(out):
        raise AlreadyTrained(f"{rid} is already trained ({out}). Use another --seed, change a setting, "
                              "or evaluate it with scripts/model/evaluate_mil.py")
    last_path, best_path = out / "checkpoints" / "last.pt", out / "checkpoints" / "best.pt"
    resuming = last_path.is_file()
    ensure_dir(out / "checkpoints")
    if not resuming:
        y_train = train_ds.labels()
        write_json(out / "run_info.json", {
            "run_id": rid, "config_hash": cfg_hash, "status": "running", "started": now_utc(),
            "git": git_info(), "environment": environment(),
            "data": {"source": data.run.source, "run": data.run.name, "manifest": str(data.run.manifest_path),
                     "manifest_sha256": data.manifest_sha256, "embedding_store": str(data.store.dir),
                     "train_volumes": len(train_ds), "val_volumes": len(val_ds),
                     "train_patients": len({r.patient_id for r in train_ds.records}),
                     "val_patients": len({r.patient_id for r in val_ds.records}),
                     "train_prevalence": dict(zip(names, np.nanmean(y_train, 0).round(4).tolist()))},
        })
        (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
        write_json(out / "labels.json", names)
        write_json(out / "store_record.json", data.store.read_record())

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    model = build_model(cfg, data.store.embed_dim, names)
    y_train = train_ds.labels()
    if cfg.features.standardize:
        model.feature_norm.fit(train_ds.bag(i) for i in range(len(train_ds)))
    if cfg.head_prior_bias:
        model.head.init_prior(np.nanmean(y_train, axis=0))
    loss_fn = LOSSES.build(cfg.loss, train_labels=y_train).to(device)
    model.to(device)

    o = cfg.optim
    optimizer = torch.optim.AdamW(_param_groups(model, o.weight_decay), lr=o.lr)
    steps_per_epoch = math.ceil(len(train_ds) / o.batch_size)
    total_steps, warmup_steps = steps_per_epoch * o.max_epochs, round(o.warmup_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, warmup_steps, total_steps))

    start_epoch, best_value, best_epoch, bad_epochs, history = 0, (math.inf if o.monitor == "val_loss" else -math.inf), 0, 0, []
    if resuming:
        state = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        start_epoch, best_value, best_epoch, bad_epochs = state["epoch"], state["best_value"], state["best_epoch"], state["bad_epochs"]
        history = state["history"]
        log(f"resuming {rid} after epoch {start_epoch}")
    log(f"{rid}: {len(train_ds)} train / {len(val_ds)} val volumes, {len(names)} labels, {steps_per_epoch} steps per epoch, "
        f"max {o.max_epochs} epochs ({total_steps} steps), device {device}")

    y_val = val_ds.labels()
    stopped = "max_epochs"
    for epoch in range(start_epoch, o.max_epochs):
        if bad_epochs >= o.patience:  # a resumed run that had already stopped
            stopped = "early_stopping"
            break
        t0 = time.perf_counter()
        train_ds.epoch = epoch
        loader = DataLoader(train_ds, batch_size=o.batch_size, shuffle=True, collate_fn=collate_bags, num_workers=0,
                            generator=torch.Generator().manual_seed(cfg.seed * 100_003 + epoch))
        model.train()
        running, seen = 0.0, 0
        for batch in loader:
            logits = model(batch["bags"].to(device), batch["mask"].to(device)).logits
            loss = loss_fn(logits, batch["labels"].to(device), batch["label_mask"].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if o.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), o.grad_clip)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(batch["volume_id"])
            seen += len(batch["volume_id"])
        probs, val_loss = predict(model, val_ds, device, loss_fn=loss_fn)
        s = M.summary(y_val, probs)
        value = {"val_macro_auroc": s["macro_auroc"], "val_macro_auprc": s["macro_auprc"], "val_loss": val_loss}[o.monitor]
        if value is None or np.isnan(value):
            raise ValueError(f"val {o.monitor} cannot be computed: no label has both classes among the "
                             f"{len(val_ds)} val volumes -- the val split is too small to select a model on")
        improved = _better(value, best_value, o.monitor)
        if improved:
            best_value, best_epoch, bad_epochs = float(value), epoch + 1, 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "monitor": o.monitor, "value": value,
                        "label_names": names, "in_dim": data.store.embed_dim, "config": cfg.to_dict(),
                        "config_hash": cfg_hash}, best_path)
        else:
            bad_epochs += 1
        history.append({
            "epoch": epoch + 1, "step": (epoch + 1) * steps_per_epoch, "lr": scheduler.get_last_lr()[0],
            "train_loss": running / max(seen, 1), "val_loss": val_loss, "val_macro_auroc": s["macro_auroc"],
            "val_macro_auprc": s["macro_auprc"], "val_micro_auroc": s["micro_auroc"], "improved": improved,
            "seconds": round(time.perf_counter() - t0, 2),
            **{f"val_auroc_{n}": float(v) for n, v in zip(names, M.auroc_per_label(y_val, probs))},
        })
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "torch_rng": torch.get_rng_state(), "epoch": epoch + 1, "best_value": best_value,
                    "best_epoch": best_epoch, "bad_epochs": bad_epochs, "history": history,
                    "label_names": names, "in_dim": data.store.embed_dim}, last_path)
        pd.DataFrame(history).to_csv(out / "metrics.csv", index=False, lineterminator="\n")
        log(f"  epoch {epoch + 1:3d}  train loss {history[-1]['train_loss']:.4f}  val loss {val_loss:.4f}  "
            f"val macro AUROC {s['macro_auroc']:.4f}  AUPRC {s['macro_auprc']:.4f}{'  *best' if improved else ''}")
        if bad_epochs >= o.patience:
            stopped = "early_stopping"
            break

    # the best epoch's model on val: predictions, thresholds (chosen here, used unchanged on test later)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model"])
    probs, val_loss = predict(model, val_ds, device, loss_fn=loss_fn)
    thresholds = M.max_f1_thresholds(y_val, probs)
    val_summary = {**M.summary(y_val, probs, thresholds), "val_loss": val_loss}
    predictions_frame(val_ds, probs, names).to_csv(out / "predictions_val.csv", index=False, lineterminator="\n")
    M.per_label_table(y_val, probs, names, thresholds).to_csv(out / "val_per_label.csv", index=False, lineterminator="\n")
    write_json(out / "thresholds.json", dict(zip(names, [None if np.isnan(t) else float(t) for t in thresholds])))
    write_json(out / "val_metrics.json", val_summary)

    info = read_json(out / "run_info.json")
    info.update(status="finished", ended=now_utc(), stopped=stopped, best_epoch=best_epoch, monitor=o.monitor,
                best_value=best_value, epochs_run=len(history), steps_run=len(history) * steps_per_epoch,
                steps_per_epoch=steps_per_epoch)
    write_json(out / "run_info.json", info)
    append_row(Path(cfg.output_dir) / TRAIN_LEDGER, {
        "run_id": rid, "name": cfg.name, "config_hash": cfg_hash, "seed": cfg.seed, "finished": info["ended"],
        "git_commit": info["git"]["commit"], "git_dirty": info["git"]["dirty"], "data_run": data.run.name,
        "embedding_store": data.store.dir.name, "aggregator": _short(cfg.aggregator), "head": _short(cfg.head),
        "loss": _short(cfg.loss), "lr": o.lr, "weight_decay": o.weight_decay, "batch_size": o.batch_size,
        "train_volumes": len(train_ds), "val_volumes": len(val_ds), "best_epoch": best_epoch,
        "epochs_run": len(history), "stopped": stopped, "val_macro_auroc": val_summary["macro_auroc"],
        "val_macro_auprc": val_summary["macro_auprc"], "val_micro_auroc": val_summary["micro_auroc"],
        "val_macro_f1": val_summary["macro_f1"], "dir": str(out),
    })
    log(f"finished {rid}: best epoch {best_epoch} ({o.monitor} {best_value:.4f}), stopped by {stopped}; results in {out}")
    return TrainResult(rid, out, best_epoch, best_value, len(history), stopped, val_summary)


def _short(spec: dict) -> str:
    """``{"type": "abmil", "branches": "single", ...}`` -> ``abmil(branches=single,...)`` for the ledger."""
    params = ",".join(f"{k}={v}" for k, v in spec.items() if k != "type")
    return f"{spec['type']}({params})" if params else spec["type"]


@dataclass
class TrainedExperiment:
    dir: Path
    cfg: ExperimentConfig
    model: VolumeClassifier
    label_names: list[str]
    thresholds: np.ndarray
    run_info: dict


def load_trained(exp_dir: str | Path, device: str | torch.device = "cpu", checkpoint: str = "best") -> TrainedExperiment:
    """A finished run folder -> its config, model (eval mode, on ``device``) and val thresholds."""
    exp_dir = Path(exp_dir)
    info = read_json(exp_dir / "run_info.json")
    if info.get("status") != "finished":
        raise FileNotFoundError(f"{exp_dir} is not a finished training run (no run_info.json with status finished)")
    with open(exp_dir / "config.yaml", encoding="utf-8") as f:
        cfg = parse_experiment(yaml.safe_load(f), origin=str(exp_dir / "config.yaml"))
    state = torch.load(exp_dir / "checkpoints" / f"{checkpoint}.pt", map_location=device, weights_only=True)
    names = state["label_names"]
    model = build_model(cfg, state["in_dim"], names)
    model.load_state_dict(state["model"])
    t = read_json(exp_dir / "thresholds.json")
    thresholds = np.array([np.nan if t.get(n) is None else t[n] for n in names], dtype=float)
    return TrainedExperiment(exp_dir, cfg, model.to(device).eval(), names, thresholds, info)
