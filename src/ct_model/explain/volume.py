"""Everything needed to explain one volume of a finished run: its stored embeddings, its HU slices,
the trained classifier, and (only for in-slice heatmaps) the stage-2 encoder that made the embeddings.

    ex = open_explainer("outputs/experiments/abmil_dale2s/1a2b3c4d/seed0")
    ev = ex.evidence("valid_1127_a_1")                  # L1, cheap, CPU is fine
    maps = ex.heatmaps("valid_1127_a_1", "emphysema", ev.top_slices("emphysema", 3))   # L2, re-encodes 3 slices
    fig = plot_label(ev, "emphysema", ex.hu("valid_1127_a_1"), maps)

The encoder is rebuilt from the embedding store's own record (the exact config and weights commit that
made the embeddings), and its fingerprint must match the store's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import parse_encoder_config
from ..data.volumes import USABLE_SPLITS, VolumeRecord
from ..training.data import ExperimentData, open_data
from ..training.trainer import TrainedExperiment, load_trained
from ..utils.device import resolve_device
from .evidence import SliceEvidence, slice_evidence
from .heatmap import SliceHeatmap, slice_heatmap


@dataclass
class VolumeExplainer:
    trained: TrainedExperiment
    data: ExperimentData
    device: object
    _records: dict = field(default_factory=dict)
    _encoder: object = None

    def _all_records(self) -> dict:
        if not self._records:
            for split in USABLE_SPLITS:
                self._records.update({r.volume_id: r for r in self.data.records(split)})
        return self._records

    def record(self, volume_id: str) -> VolumeRecord:
        records = self._all_records()
        if volume_id not in records:
            raise KeyError(f"{volume_id} is not a usable volume of run {self.data.run.name!r}")
        return records[volume_id]

    def volume_ids(self, split: str | None = None) -> list[str]:
        return [v for v, r in self._all_records().items() if split is None or r.split == split]

    def bag(self, volume_id: str) -> np.ndarray:
        rec = self.record(volume_id)
        if not self.data.store.has(volume_id, rec.n_slices):
            raise FileNotFoundError(f"{volume_id} has no embeddings in {self.data.store.dir}")
        return np.asarray(self.data.store.load(volume_id), dtype=np.float32)

    def hu(self, volume_id: str) -> np.ndarray:
        """(n_slices, H, W) int16 HU from the cache, memory-mapped (only the slices used are read)."""
        return np.load(self.record(volume_id).npy_path, mmap_mode="r")

    def labels(self, volume_id: str) -> np.ndarray:
        return np.asarray(self.record(volume_id).labels, dtype=float)[self.data.label_index]

    def evidence(self, volume_id: str) -> SliceEvidence:
        return slice_evidence(self.trained.model, self.bag(volume_id), self.trained.thresholds, self.device)

    def encoder(self):
        if self._encoder is None:
            from ..encoders import build_encoder

            record = self.data.store.read_record()
            enc_cfg = parse_encoder_config(record["encoder_config"], origin=str(self.data.store.dir))
            if enc_cfg.fingerprint() != record.get("encoder_fingerprint"):
                raise RuntimeError(f"the encoder config in {self.data.store.dir} does not match its fingerprint")
            if enc_cfg.weights.source == "none":
                raise RuntimeError(f"the embeddings in {self.data.store.dir} were made by a randomly initialised encoder "
                                   "(weights.source: none), which cannot be rebuilt -- no in-slice heatmaps")
            self._encoder = build_encoder(enc_cfg).to(self.device)
        return self._encoder

    def heatmaps(self, volume_id: str, label: str, slice_indices, min_cosine: float = 0.99) -> list[SliceHeatmap]:
        bag, hu = self.bag(volume_id), self.hu(volume_id)
        return [slice_heatmap(self.trained.model, self.encoder(), bag, np.array(hu[int(k)]), int(k), label,
                              self.device, min_cosine) for k in slice_indices]


def open_explainer(exp_dir: str | Path, device: str = "cpu", source: str | None = None,
                   run: str | None = None) -> VolumeExplainer:
    dev = resolve_device(device)
    trained = load_trained(exp_dir, device=dev)
    data = open_data(trained.cfg, source=source, run=run, labels=trained.label_names)
    return VolumeExplainer(trained, data, dev)


def plot_label(ev: SliceEvidence, label: str, hu_volume: np.ndarray, heatmaps: list[SliceHeatmap] | None = None,
               k: int = 5, true_value: float | None = None, window=(-1000, 400)):
    """One figure: the label's per-slice contribution along z, then its top-``k`` slices (lung window),
    with heatmap overlays when given."""
    import matplotlib.pyplot as plt

    c = ev.label_names.index(label)
    top = ev.top_slices(c, k)
    by_slice = {h.slice_index: h for h in (heatmaps or [])}
    fig = plt.figure(figsize=(3.2 * k, 6.2))
    grid = fig.add_gridspec(2, k, height_ratios=[1, 1.6])
    ax = fig.add_subplot(grid[0, :])
    contrib = ev.contributions[c]
    ax.bar(np.arange(len(contrib)), contrib, width=1.0, color=np.where(contrib >= 0, "tab:red", "tab:blue"))
    ax.scatter(top, contrib[top], color="black", zorder=3, s=12)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("slice (head -> foot, preprocessed volume)")
    ax.set_ylabel("contribution to logit")
    thr = ev.thresholds[c]
    title = f"{label}: p = {ev.probs[c]:.3f}" + (f" (val threshold {thr:.3f} -> {'positive' if ev.predicted[c] else 'negative'})" if not np.isnan(thr) else "")
    if true_value is not None and not np.isnan(true_value):
        title += f"; label in manifest: {int(true_value)}"
    ax.set_title(title)
    for j, s in enumerate(top):
        a = fig.add_subplot(grid[1, j])
        a.imshow(np.asarray(hu_volume[int(s)]), cmap="gray", vmin=window[0], vmax=window[1])
        if int(s) in by_slice:
            a.imshow(by_slice[int(s)].heatmap, cmap="jet", alpha=0.35, vmin=0, vmax=1)
        a.set_title(f"slice {int(s)}: {contrib[s]:+.3f}", fontsize=9)
        a.axis("off")
    fig.tight_layout()
    return fig
