"""Several runs used together (data.run: "a+b"): one manifest, no leak, a stable config hash."""
from pathlib import Path

import pandas as pd
import pytest

from ct_model.data.volumes import CombinedRun, combine_manifests, load_run_manifest, manifest_sha256
from ct_model.training.config import load_experiment, override
from ct_model.training.data import open_data
from ct_model.training.records import file_sha256
from ct_preprocessing.config import load_config


def _soft_run(mil_project, patients, split_of=None):
    """A second run holding soft reconstructions of some synth patients (plus the shared test pool)."""
    runs = mil_project["exp"].parent / "data" / "runs" / "ctrate"
    synth = pd.read_csv(runs / "synth" / "manifest.csv")
    soft = synth[synth.patient_id.isin(patients) & (synth.kernel_class == "sharp")].copy()
    soft["volume_id"] = soft.volume_id + "_soft"
    soft["kernel_class"] = "soft"
    if split_of:
        soft["split"] = soft.patient_id.map(split_of).fillna(soft.split)
    (runs / "soft").mkdir()
    pd.concat([soft, synth[synth.split == "test"]]).to_csv(runs / "soft" / "manifest.csv", index=False)
    return soft


def test_combined_run_keeps_shared_volumes_once_and_hashes_its_parts(mil_project):
    import numpy as np
    from ct_model.config import load_encoder_config
    from ct_model.embeddings.store import open_store

    soft = _soft_run(mil_project, ["train_0", "train_1", "val_0"])
    cfg = override(load_experiment(str(mil_project["exp"])), **{"data.run": "synth+soft", "data.train_kernel_classes": None})
    data_cfg = load_config(cfg.data_config)
    store = open_store(Path(data_cfg.paths.runs_dir).parent / "embeddings", load_encoder_config(cfg.encoder),
                       hu_cache_fingerprint=None, embed_dim=16)
    for vid in soft.volume_id:  # the soft volumes need embeddings like any other
        store.write(vid, np.zeros((int(soft.set_index("volume_id").n_slices[vid]), 16)))

    run, manifest = load_run_manifest(data_cfg, "ctrate", "synth+soft")
    assert isinstance(run, CombinedRun) and run.name == "synth+soft"
    synth = pd.read_csv(run.parts[0].manifest_path)
    assert len(manifest) == len(synth) + len(soft)  # the test pool both runs list is kept once
    assert manifest_sha256(run) != file_sha256(run.parts[0].manifest_path)
    assert manifest_sha256(run.parts[0]) == file_sha256(run.parts[0].manifest_path)  # single runs: unchanged hash

    data = open_data(cfg)
    assert data.run.name == "synth+soft"
    val = data.records("val")
    assert {r.kernel_class for r in val} == {"sharp", "soft"}  # val now holds soft volumes too


def test_a_patient_with_two_splits_is_refused(mil_project):
    _soft_run(mil_project, ["train_0"], split_of={"train_0": "val"})
    with pytest.raises(ValueError, match="different splits"):
        load_run_manifest(load_config(load_experiment(str(mil_project["exp"])).data_config), "ctrate", "synth+soft")


def test_runs_with_other_labels_cannot_be_combined():
    a = pd.DataFrame({"volume_id": ["v1"], "patient_id": ["p1"], "split": ["train"], "label_x": [1]})
    b = pd.DataFrame({"volume_id": ["v2"], "patient_id": ["p2"], "split": ["train"], "label_y": [1]})
    with pytest.raises(ValueError, match="other label columns"):
        combine_manifests([a, b], ["a", "b"])
