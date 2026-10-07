"""The data an experiment uses: a run's manifest (splits, labels), joined to the encoder's embedding store.

Opening checks, before anything trains:
  * the embedding store exists and still matches the HU cache (open_store(create=False));
  * the requested labels are columns of the manifest;
  * every selected volume has embeddings (EmbeddingBagDataset lists any that do not).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ct_preprocessing.cache_record import read_cache_fingerprint
from ct_preprocessing.config import load_config
from ct_preprocessing.runs import Run

from ..config import load_encoder_config, load_stage2_config
from ..data.datasets import EmbeddingBagDataset
from ..data.slices import SliceSampler
from ..data.volumes import VolumeRecord, label_names, load_run_manifest, select_volumes
from ..embeddings.store import EmbeddingStore, open_store
from .config import ExperimentConfig
from .records import file_sha256


@dataclass
class ExperimentData:
    run: Run
    manifest: pd.DataFrame
    manifest_sha256: str
    cache_dir: str
    store: EmbeddingStore
    label_names: list[str]
    label_index: list[int]
    qc_passed_only: bool

    def records(self, split: str, kernel_classes: tuple[str, ...] | None = None) -> list[VolumeRecord]:
        return select_volumes(self.manifest, self.cache_dir, splits=(split,),
                              qc_passed_only=self.qc_passed_only, kernel_classes=kernel_classes)

    def dataset(self, split: str, kernel_classes: tuple[str, ...] | None = None, *, preload: bool = True,
                sampler: SliceSampler | None = None, seed: int = 0) -> EmbeddingBagDataset:
        records = self.records(split, kernel_classes)
        if not records:
            raise ValueError(f"run {self.run.name!r} has no {split} volumes with kernel classes {kernel_classes}")
        return EmbeddingBagDataset(records, self.store, sampler=sampler, seed=seed, preload=preload,
                                   label_index=self.label_index)


def open_data(cfg: ExperimentConfig, *, source: str | None = None, run: str | None = None,
              labels: list[str] | None = None) -> ExperimentData:
    """``source`` / ``run`` default to the experiment's; ``labels`` to the experiment's (or all)."""
    stage2 = load_stage2_config(cfg.encode_config)
    data_cfg = load_config(cfg.data_config)
    enc_cfg = load_encoder_config(cfg.encoder)
    resolved, manifest = load_run_manifest(data_cfg, source or cfg.data.source, run or cfg.data.run)
    store = open_store(stage2.encode.embeddings_dir, enc_cfg,
                       hu_cache_fingerprint=read_cache_fingerprint(data_cfg.paths.cache_dir), create=False)
    available = label_names(manifest)
    wanted = list(labels or cfg.data.labels or available)
    missing = [name for name in wanted if name not in available]
    if missing:
        raise ValueError(f"labels {missing} are not columns of {resolved.manifest_path} (it has: {available})")
    if not wanted:
        raise ValueError(f"{resolved.manifest_path} has no label_* columns -- merge the labels first")
    return ExperimentData(
        run=resolved, manifest=manifest, manifest_sha256=file_sha256(resolved.manifest_path),
        cache_dir=data_cfg.paths.cache_dir, store=store, label_names=wanted,
        label_index=[available.index(name) for name in wanted], qc_passed_only=cfg.data.qc_passed_only,
    )
