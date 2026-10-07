"""Shared fixtures for the model-side tests: a tiny HU cache + run manifest on disk, no download needed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

LABELS = ["lung_nodule", "emphysema"]


def write_volume(cache_dir: Path, volume_id: str, n_slices: int, size: int = 32, seed: int = 0) -> np.ndarray:
    """An int16 HU volume whose slice i has mean HU ~ -1000 + 10 * i, so slice order is checkable."""
    rng = np.random.default_rng(seed)
    base = (-1000 + 10 * np.arange(n_slices))[:, None, None]
    hu = (base + rng.normal(0, 5, size=(n_slices, size, size))).astype(np.int16)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{volume_id}.npy", hu)
    return hu


@pytest.fixture
def mini_project(tmp_path: Path) -> dict:
    """data/cache with 5 volumes, a run 'r1' of source ctrate with its manifest, and the config files
    pointing at them. Returns the paths."""
    data = tmp_path / "data"
    cache = data / "cache"
    rows = [
        # volume_id, patient, split, qc, n_slices, kernel, labels
        ("train_1_a_1", "train_1", "train", True, 12, "sharp", (1, 0)),
        ("train_2_a_1", "train_2", "train", True, 9, "sharp", (0, "")),
        ("train_3_a_1", "train_3", "val", True, 10, "sharp", (0, 1)),
        ("valid_1_a_1", "valid_1", "test", True, 11, "sharp", (1, 1)),
        ("valid_1_a_2", "valid_1", "test", True, 11, "soft", (1, 1)),
        ("train_4_a_1", "train_4", "excluded", False, 8, "sharp", (0, 0)),
    ]
    records = []
    for i, (vid, pid, split, qc, n, kc, labels) in enumerate(rows):
        write_volume(cache, vid, n, seed=i)
        records.append({
            "volume_id": vid, "patient_id": pid, "split": split, "qc_passed": qc, "n_slices": n,
            "kernel_class": kc, "npy_path": f"data\\cache\\{vid}.npy",  # Windows separators, as merge writes them
            **{f"label_{name}": v for name, v in zip(LABELS, labels)},
        })
    run_dir = data / "runs" / "ctrate" / "r1"
    run_dir.mkdir(parents=True)
    pd.DataFrame(records).to_csv(run_dir / "manifest.csv", index=False)
    (cache / ".cache_fingerprint.json").write_text(json.dumps({"fingerprint": "hucachefp0000001"}), encoding="utf-8")

    pre_cfg = tmp_path / "preprocessing.yaml"
    pre_cfg.write_text(
        "paths:\n"
        f"  cache_dir: {cache.as_posix()}\n"
        f"  runs_dir: {(data / 'runs').as_posix()}\n"
        "preprocess: {}\nqc: {}\n",
        encoding="utf-8",
    )
    enc_cfg = tmp_path / "tiny_vit.yaml"
    enc_cfg.write_text(
        "name: tiny_vit\n"
        "type: timm_vit\n"
        "arch: vit_tiny_patch16_224\n"
        "model_args: {img_size: 64, in_chans: 1, num_classes: 0, dynamic_img_size: true}\n"
        "embed_dim: 192\n"
        "weights: {source: none}\n"
        "input: {transform: clip_zscore, clip_hu: [-997.0, 888.0], mean_hu: -142.39, std_hu: 360.97, size_hw: [32, 32]}\n"
        "pooling: cls\n",
        encoding="utf-8",
    )
    stage2 = tmp_path / "encode.yaml"
    stage2.write_text(
        f"data_config: {pre_cfg.as_posix()}\n"
        f"encoder: {enc_cfg.as_posix()}\n"
        "selection: {source: ctrate, run: r1}\n"
        f"encode: {{embeddings_dir: {(data / 'embeddings').as_posix()}, device: cpu, slice_batch_size: 4, num_workers: 0, min_free_gb: 0}}\n",
        encoding="utf-8",
    )
    return {"root": tmp_path, "data": data, "cache": cache, "run_dir": run_dir, "pre_cfg": pre_cfg,
            "enc_cfg": enc_cfg, "stage2": stage2, "embeddings": data / "embeddings"}
