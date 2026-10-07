"""scripts/model/encode_volumes.py and check_encoder.py, run as a user would (tiny random ViT, CPU)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "model"


def call(monkeypatch, name: str, *args) -> int:
    spec = importlib.util.spec_from_file_location(f"model_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", [f"{name}.py", *map(str, args)])
    return mod.main()


def test_encode_dry_run_then_real_then_resume(mini_project, monkeypatch, capsys):
    cfg = mini_project["stage2"]
    assert call(monkeypatch, "encode_volumes", "--config", cfg, "--dry-run") == 0
    out = capsys.readouterr().out
    assert out.startswith("encode_volumes: encoder=tiny_vit") and "selected 5 volumes (53 slices" in out
    assert not mini_project["embeddings"].exists()  # dry run writes nothing

    assert call(monkeypatch, "encode_volumes", "--config", cfg) == 0
    out = capsys.readouterr().out
    assert "done: 5 encoded, 0 already in the store, 0 failed" in out
    (store_dir,) = mini_project["embeddings"].iterdir()
    assert store_dir.name.startswith("tiny_vit-")
    emb = np.load(store_dir / "train_1_a_1.npy")
    assert emb.shape == (12, 192) and emb.dtype == np.float16 and np.isfinite(emb).all()
    assert (store_dir / "index.csv").read_text().count("\n") == 6

    assert call(monkeypatch, "encode_volumes", "--config", cfg, "--kernel-classes", "soft", "--splits", "test") == 0
    assert "done: all 1 selected volumes are already in the store" in capsys.readouterr().out
    assert call(monkeypatch, "encode_volumes", "--config", cfg, "--dry-run") == 0
    assert "dry run: 5 already encoded, 0 to encode" in capsys.readouterr().out


def test_low_disk_refused_before_loading_the_model(mini_project, monkeypatch, capsys):
    assert call(monkeypatch, "encode_volumes", "--config", mini_project["stage2"], "--min-free-gb", "1e12") == 1
    assert "below min_free_gb" in capsys.readouterr().err
    assert not any(mini_project["embeddings"].glob("*"))  # no store was created


def test_missing_cache_file_is_an_error(mini_project, monkeypatch, capsys):
    (mini_project["cache"] / "train_3_a_1.npy").unlink()
    assert call(monkeypatch, "encode_volumes", "--config", mini_project["stage2"]) == 1
    assert "train_3_a_1" in capsys.readouterr().err


def test_check_encoder_writes_nothing(mini_project, monkeypatch, capsys):
    assert call(monkeypatch, "check_encoder", "--config", mini_project["stage2"], "--max-slices", "6") == 0
    out = capsys.readouterr().out
    assert "output (6, 192)" in out and "finite: True" in out and "projected: 5 selected volumes" in out
    assert not mini_project["embeddings"].exists()
