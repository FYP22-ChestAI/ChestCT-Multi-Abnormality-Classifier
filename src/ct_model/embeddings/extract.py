"""Stage 2's loop: every selected volume of the HU cache -> its per-slice embeddings in the store.

    cache .npy (int16, mmap) --DataLoader--> device --encoder (transform + ViT)--> (n_slices, D) --> store

* Resumable: a volume whose embeddings are already in the store with the right shape is skipped, so
  an interrupted run (or a second run sharing volumes) continues where it stopped.
* Memory: each volume is split into ``slice_batch_size`` slices per forward pass, under
  ``torch.inference_mode`` and (on the GPU) bfloat16 autocast. On CUDA out-of-memory the batch halves
  and the same slices are retried; the smaller batch is kept for the rest of the run.
* Disk: below ``min_free_gb`` free on the store's filesystem the run stops cleanly (nothing half-written:
  every file is written atomically).
* One bad volume never stops the run: it is recorded in the summary with its reason. But if the first
  volumes ALL fail, something systematic is wrong (weights, device, config) and the run aborts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from ct_preprocessing.ingest.engine import free_gb

from ..data.datasets import HUVolumeDataset
from ..data.volumes import VolumeRecord
from .store import EmbeddingStore

_OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
_ABORT_AFTER = 5  # consecutive failures before anything succeeded -> abort


@dataclass
class EncodeSummary:
    selected: int = 0
    skipped: int = 0  # already in the store
    encoded: int = 0
    slices: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0
    slice_batch_size: int = 0  # the batch size in use at the end (after any OOM back-off)
    peak_vram_gb: float | None = None
    stopped: str | None = None  # why the run stopped early, if it did

    @property
    def slices_per_second(self) -> float:
        return self.slices / self.seconds if self.seconds > 0 else 0.0


def encode_slices(
    encoder: torch.nn.Module,
    hu: torch.Tensor,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    slice_batch_size: int,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, int]:
    """(K, H, W) HU slices -> ((K, D) float32 embeddings, the slice batch size that fitted)."""
    out: list[torch.Tensor] = []
    bs, i = slice_batch_size, 0
    while i < hu.shape[0]:
        chunk = hu[i:i + bs].to(device, non_blocking=True)
        try:
            with torch.inference_mode(), torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                emb = encoder(chunk)
        except _OOM:
            if bs == 1:
                raise
            del chunk
            if device.type == "cuda":
                torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            log(f"  out of memory -- slice_batch_size lowered to {bs}")
            continue
        out.append(emb.float().cpu())
        i += chunk.shape[0]
    return torch.cat(out).numpy(), bs


def encode_volumes(
    records: list[VolumeRecord],
    encoder: torch.nn.Module,
    store: EmbeddingStore,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    slice_batch_size: int = 128,
    num_workers: int = 2,
    min_free_gb: float = 0.0,
    log: Callable[[str], None] = print,
    log_every: int = 25,
) -> EncodeSummary:
    summary = EncodeSummary(selected=len(records), slice_batch_size=slice_batch_size)
    todo = [r for r in records if not store.has(r.volume_id, r.n_slices)]
    summary.skipped = len(records) - len(todo)
    if not todo:
        return summary

    encoder = encoder.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loader = DataLoader(
        HUVolumeDataset(todo), batch_size=None, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    start = time.perf_counter()
    for n, item in enumerate(loader, start=1):
        free = free_gb(store.dir) if min_free_gb else 0.0
        if min_free_gb and free < min_free_gb:
            summary.stopped = f"only {free:.1f} GB free on {store.dir}, below min_free_gb {min_free_gb}"
            break
        vid = item["volume_id"]
        try:
            if item["error"]:
                raise RuntimeError(item["error"])
            emb, summary.slice_batch_size = encode_slices(
                encoder, item["hu"], device=device, amp_dtype=amp_dtype,
                slice_batch_size=summary.slice_batch_size, log=log,
            )
            store.write(vid, emb)
            summary.encoded += 1
            summary.slices += emb.shape[0]
        except Exception as exc:  # noqa: BLE001 - one bad volume never stops the run
            summary.failed[vid] = f"{type(exc).__name__}: {exc}"[:500]
            log(f"  failed {vid}: {summary.failed[vid]}")
            if summary.encoded == 0 and len(summary.failed) >= _ABORT_AFTER:
                raise RuntimeError(
                    f"the first {len(summary.failed)} volumes all failed -- aborting. First error: "
                    f"{next(iter(summary.failed.values()))}"
                ) from exc
        if n % log_every == 0 or n == len(todo):
            elapsed = time.perf_counter() - start
            rate = summary.slices / elapsed if elapsed > 0 else 0.0
            eta = (len(todo) - n) * elapsed / n
            log(f"  {n}/{len(todo)} volumes, {rate:.0f} slices/s, {len(summary.failed)} failed, eta {eta / 60:.1f} min")
    summary.seconds = time.perf_counter() - start
    if device.type == "cuda":
        summary.peak_vram_gb = torch.cuda.max_memory_allocated(device) / 2**30
    return summary
