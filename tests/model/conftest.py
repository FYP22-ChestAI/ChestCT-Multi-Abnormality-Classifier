"""Shared fixtures for the model-side tests: a tiny HU cache + run manifest on disk, no download needed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

LABELS = ["lung_nodule", "emphysema"]
MIL_LABELS = ["nodule", "effusion", "emphysema"]
MIL_DIM = 16
REPO = Path(__file__).resolve().parents[2]


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


@pytest.fixture
def mil_project(tmp_path):
    """A synthetic embedding store with PLANTED findings (see test_train_e2e.py) and a ready experiment
    config: label c is present iff 3 consecutive slices carry +3 along dimension c."""
    from ct_model.config import load_encoder_config
    from ct_model.embeddings.store import open_store

    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    (data / "cache").mkdir(parents=True)
    run_dir = data / "runs" / "ctrate" / "synth"
    run_dir.mkdir(parents=True)
    enc_yaml = tmp_path / "synth_enc.yaml"
    enc_yaml.write_text(
        f"name: synth\narch: none\nembed_dim: {MIL_DIM}\n"
        "input: {transform: clip_zscore, clip_hu: [-1000, 1000], mean_hu: 0, std_hu: 1}\n", encoding="utf-8")
    store = open_store(data / "embeddings", load_encoder_config(str(enc_yaml)), hu_cache_fingerprint=None, embed_dim=MIL_DIM)

    rows, planted = [], {}
    plan = [("train", 160, ["sharp"]), ("val", 60, ["sharp"]), ("test", 40, ["sharp", "soft"])]
    for split, n_patients, kernels in plan:
        for i in range(n_patients):
            pid = f"{split}_{i}"
            y = (rng.random(3) < [0.45, 0.25, 0.12]).astype(int)  # imbalanced, like CT-RATE
            n = int(rng.integers(20, 41))
            base = rng.normal(0, 1, size=(n, MIL_DIM))
            for c in np.flatnonzero(y):
                start = int(rng.integers(0, n - 3))
                base[start:start + 3, c] += 3.0
                planted[(pid, c)] = set(range(start, start + 3))
            for k, kernel in enumerate(kernels):
                vid = f"{pid}_r{k}"
                store.write(vid, base + rng.normal(0, 0.05, size=base.shape))
                rows.append({"volume_id": vid, "patient_id": pid, "split": split, "qc_passed": True, "n_slices": n,
                             "kernel_class": kernel, "manufacturer": ["Philips", "Siemens"][i % 2],
                             **{f"label_{name}": int(v) for name, v in zip(MIL_LABELS, y)}})
    pd.DataFrame(rows).to_csv(run_dir / "manifest.csv", index=False)

    pre = tmp_path / "preprocessing.yaml"
    pre.write_text(f"paths:\n  cache_dir: {(data / 'cache').as_posix()}\n  runs_dir: {(data / 'runs').as_posix()}\n"
                   "preprocess: {}\nqc: {}\n", encoding="utf-8")
    enc = tmp_path / "encode.yaml"
    enc.write_text(f"data_config: {pre.as_posix()}\nencode: {{embeddings_dir: {(data / 'embeddings').as_posix()}}}\n", encoding="utf-8")
    exp = yaml.safe_load((REPO / "configs" / "model" / "experiments" / "abmil_dale2s.yaml").read_text(encoding="utf-8"))
    exp.update(name="synth_abmil", data_config=pre.as_posix(), encode_config=enc.as_posix(), encoder=enc_yaml.as_posix(),
               device="cpu", output_dir=(tmp_path / "outputs").as_posix())
    exp["data"]["run"] = "synth"
    # per-label attention: on these data (each finding in its own slices) one shared attention cannot
    # cover three findings at once -- measured: single ~0.82 vs per_label ~0.95 val macro AUROC
    exp["aggregator"].update(hidden_dim=16, attn_dim=8, dropout=0.0, branches="per_label")
    exp["optim"].update(lr=3e-3, batch_size=16, max_epochs=40, warmup_epochs=1, patience=10)
    exp_path = tmp_path / "synth_abmil.yaml"
    exp_path.write_text(yaml.safe_dump(exp), encoding="utf-8")
    return {"exp": exp_path, "outputs": tmp_path / "outputs", "planted": planted, "labels": MIL_LABELS}
