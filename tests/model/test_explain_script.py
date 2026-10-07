"""scripts/model/explain_volume.py end to end: tiny ViT embeddings made by encode_volumes.py, a short
training run, then L1 + L2 evidence for a test volume -- the encoder is rebuilt from the store's record."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("timm")

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "model"


def call(monkeypatch, name, *args):
    spec = importlib.util.spec_from_file_location(f"model_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", [f"{name}.py", *map(str, args)])
    return mod.main()


def _train_tiny(mini_project, monkeypatch, tmp_path, local_weights: bool):
    """Encode the mini project with the tiny ViT and train 2 epochs; returns the run folder."""
    # two val volumes with both classes of each label, so val metrics exist
    manifest = mini_project["run_dir"] / "manifest.csv"
    m = pd.read_csv(manifest)
    m.loc[m.volume_id == "train_2_a_1", ["split", "label_lung_nodule", "label_emphysema"]] = ["val", 1, 0]
    m.to_csv(manifest, index=False)
    if local_weights:  # fixed weights from a local file, so the encoder can be rebuilt exactly
        import timm
        from safetensors.torch import save_file

        vit = timm.create_model("vit_tiny_patch16_224", img_size=64, in_chans=1, num_classes=0, dynamic_img_size=True)
        weights = tmp_path / "tiny.safetensors"
        save_file({k: v.contiguous() for k, v in vit.state_dict().items()}, str(weights))
        enc_yaml = mini_project["enc_cfg"]
        text = enc_yaml.read_text(encoding="utf-8").replace(
            "weights: {source: none}", f"weights: {{source: local_safetensors, filename: '{weights.as_posix()}'}}")
        enc_yaml.write_text(text, encoding="utf-8")
    assert call(monkeypatch, "encode_volumes", "--config", mini_project["stage2"]) == 0

    exp = yaml.safe_load((REPO / "configs/model/experiments/abmil_dale2s.yaml").read_text(encoding="utf-8"))
    exp.update(name="tiny_exp", data_config=mini_project["pre_cfg"].as_posix(), encode_config=mini_project["stage2"].as_posix(),
               encoder=mini_project["enc_cfg"].as_posix(), device="cpu", output_dir=(tmp_path / "out").as_posix())
    exp["data"].update(run="r1")
    exp["aggregator"].update(hidden_dim=8, attn_dim=4)
    exp["optim"].update(max_epochs=2, warmup_epochs=0)
    path = tmp_path / "tiny_exp.yaml"
    path.write_text(yaml.safe_dump(exp), encoding="utf-8")
    assert call(monkeypatch, "train_mil", "--experiment", path) == 0
    (run,) = (tmp_path / "out" / "tiny_exp").glob("*/seed0")
    return run


def test_explain_volume_with_heatmaps(mini_project, monkeypatch, capsys, tmp_path):
    run = _train_tiny(mini_project, monkeypatch, tmp_path, local_weights=True)
    capsys.readouterr()

    assert call(monkeypatch, "explain_volume", "--experiment-dir", run, "--volume-id", "valid_1_a_1",
                "--labels", "emphysema", "lung_nodule", "--top-k", "3", "--heatmaps", "--min-cosine", "0.999",
                "--device", "cpu") == 0
    out = run / "explain" / "valid_1_a_1"
    assert (out / "emphysema.png").is_file() and (out / "lung_nodule.png").is_file()
    report = json.loads((out / "evidence.json").read_text())
    em = report["labels"]["emphysema"]
    assert len(em["top_slices"]) == 3 and report["slices"] == 11 and em["manifest_label"] == 1
    assert all(c > 0.999 for c in em["reencode_cosine"])  # the store's own encoder was rebuilt
    contributions = [s["contribution"] for s in em["top_slices"]]
    assert contributions == sorted(contributions, reverse=True)

    assert call(monkeypatch, "explain_volume", "--experiment-dir", run, "--volume-id", "valid_1_a_1",
                "--labels", "nope") == 1
    assert "unknown labels" in capsys.readouterr().err


def test_random_init_store_gets_no_heatmaps(mini_project, monkeypatch, capsys, tmp_path):
    run = _train_tiny(mini_project, monkeypatch, tmp_path, local_weights=False)
    capsys.readouterr()
    assert call(monkeypatch, "explain_volume", "--experiment-dir", run, "--volume-id", "valid_1_a_1",
                "--labels", "emphysema", "--heatmaps", "--device", "cpu") == 1
    assert "randomly initialised" in capsys.readouterr().err
    # slice-level evidence (L1) needs no encoder, so it still works
    assert call(monkeypatch, "explain_volume", "--experiment-dir", run, "--volume-id", "valid_1_a_1",
                "--labels", "emphysema", "--device", "cpu") == 0
