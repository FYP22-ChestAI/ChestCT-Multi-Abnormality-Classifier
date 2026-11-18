"""Phase 2: LoRA on a timm ViT, adapter files, pair alignment, augmentation, and an end-to-end adaptation run."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from ct_model.adapt.augment import blur, kernel_augment, sharpen  # noqa: E402
from ct_model.adapt.config import parse_adapt_config  # noqa: E402
from ct_model.adapt.data import AdaptDataset, align_offset, scan_items  # noqa: E402
from ct_model.config import parse_encoder_config  # noqa: E402
from ct_model.data.slices import SliceSampler  # noqa: E402
from ct_model.data.volumes import select_volumes  # noqa: E402
from ct_model.encoders.lora import (  # noqa: E402
    LoRALinear, forward_features_split, load_adapter, lora_parameters, save_adapter,
)
from ct_model.encoders.registry import build_encoder  # noqa: E402

TINY = {"name": "tiny", "type": "timm_vit", "arch": "vit_tiny_patch16_224",
        "model_args": {"img_size": 32, "in_chans": 1, "num_classes": 0, "dynamic_img_size": True},
        "embed_dim": 192, "weights": {"source": "none"},
        "input": {"transform": "clip_zscore", "clip_hu": [-997.0, 888.0], "mean_hu": -142.39, "std_hu": 360.97,
                  "size_hw": [32, 32]}, "pooling": "cls"}
LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["qkv", "proj"], "last_n_blocks": 4}


def _encoder(lora=LORA, frozen=False, **extra):
    torch.manual_seed(0)
    return build_encoder(parse_encoder_config({**TINY, **extra, "lora": dict(lora) if lora else None}), frozen=frozen)


def test_lora_starts_at_the_base_model_and_trains_only_its_matrices():
    base, adapted = _encoder(lora=None), _encoder()
    hu = torch.randint(-1000, 500, (3, 32, 32)).float()
    assert torch.allclose(base(hu), adapted(hu), atol=1e-5)  # B = 0: exactly the frozen encoder
    trainable = {n for n, p in adapted.named_parameters() if p.requires_grad}
    assert trainable == set(lora_parameters(adapted)) and len(trainable) == 4 * 2 * 2  # 4 blocks x (qkv, proj) x (A, B)
    wrapped = [m for m in adapted.model.blocks.modules() if isinstance(m, LoRALinear)]
    assert len(wrapped) == 8 and adapted.lora_first_block == len(adapted.model.blocks) - 4


def test_split_forward_matches_timm_and_carries_gradients_only_to_lora():
    enc = _encoder()
    x = enc.input_transform(torch.randint(-1000, 500, (2, 32, 32)).float())
    for p in lora_parameters(enc).values():
        torch.nn.init.normal_(p, std=0.02)
    with torch.no_grad():
        ref = enc.model.forward_features(x)
    out = forward_features_split(enc.model, x, enc.lora_first_block, checkpoint=True)
    assert torch.allclose(ref, out, atol=1e-5)
    enc.pool(out).sum().backward()
    grads = {n for n, p in enc.named_parameters() if p.grad is not None}
    assert grads and grads <= set(lora_parameters(enc))


def test_adapter_round_trip_and_frozen_encoding(tmp_path):
    enc = _encoder()
    for p in lora_parameters(enc).values():
        torch.nn.init.normal_(p, std=0.05)
    path = save_adapter(enc, tmp_path / "adapter.safetensors", {"epoch": 3})
    hu = torch.randint(-1000, 500, (2, 32, 32)).float()
    loaded = _encoder(lora={**LORA, "adapter": str(path)}, frozen=True)
    assert not any(p.requires_grad for p in loaded.parameters())  # encoding: frozen, LoRA included
    assert torch.allclose(enc(hu), loaded(hu), atol=1e-5)
    assert loaded.provenance()["lora_adapter_sha256"]
    with pytest.raises(RuntimeError, match="does not match"):
        load_adapter(_encoder(lora={**LORA, "last_n_blocks": 2}), path)
    # the adapter path is part of the config, so an adapted encoder gets its own store folder
    a = parse_encoder_config({**TINY, "lora": {**LORA, "adapter": str(path)}})
    assert a.store_name != parse_encoder_config(TINY).store_name


def test_pair_alignment_finds_the_offset():
    rng = np.random.default_rng(0)
    profile = np.cumsum(rng.normal(size=60))
    off, ok = align_offset(profile[2:], profile[:-1] + rng.normal(0, 0.01, 59), 2)
    assert ok and off == 2   # slice i of A (= original i + 2) is slice i + 2 of B (= original i + 2)
    assert align_offset(profile, profile[:50], 2) == (0, False)  # 10 slices apart: not aligned


def test_kernel_augmentation_moves_texture_not_mean():
    x = torch.randn(2, 32, 32) * 100
    b, s = blur(x, 1.0), sharpen(x, 1.0)
    assert b.shape == x.shape and b.std() < x.std() < s.std()
    assert abs(float(b.mean() - x.mean())) < 5
    assert kernel_augment(x, "blur", torch.Generator().manual_seed(0), (0.5, 1.2), (0.5, 1.5)).shape == x.shape


# ------------------------------------------------------------------ an adaptation run on a tiny project
@pytest.fixture
def adapt_project(tmp_path):
    rng = np.random.default_rng(0)
    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    rows = []
    for split, n in (("train", 10), ("val", 8)):
        for i in range(n):
            pid = f"{split}_{i}"
            y = [int(i % 2), int(i % 3 == 0)]
            base = (-800 + 30 * np.arange(14))[:, None, None] + rng.normal(0, 40, (14, 32, 32)) + 400 * y[0]
            for r, kc in enumerate(["sharp", "soft"] if i < 5 else ["sharp"]):
                vid = f"{pid}_a_{r + 1}"
                vol = base if kc == "sharp" else blur(torch.from_numpy(base).float(), 1.0).numpy()
                np.save(cache / f"{vid}.npy", vol.astype(np.int16))
                rows.append({"volume_id": vid, "patient_id": pid, "scan_id": "a", "split": split, "qc_passed": True,
                             "n_slices": 14, "kernel_class": kc, "label_a": y[0], "label_b": y[1]})
    run = tmp_path / "data" / "runs" / "ctrate" / "r1"
    run.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(run / "manifest.csv", index=False)
    (cache / ".cache_fingerprint.json").write_text('{"fingerprint": "x"}')
    pre = tmp_path / "preprocessing.yaml"
    pre.write_text(f"paths:\n  cache_dir: {cache.as_posix()}\n  runs_dir: {(tmp_path / 'data' / 'runs').as_posix()}\n"
                   "preprocess: {}\nqc: {}\n")
    enc = tmp_path / "tiny.yaml"
    enc.write_text(yaml.safe_dump(TINY))
    return {"root": tmp_path, "pre": pre, "enc": enc, "rows": pd.DataFrame(rows)}


def test_scan_items_pair_the_two_kernels(adapt_project):
    m = adapt_project["rows"]
    records = select_volumes(m, adapt_project["root"] / "data" / "cache", splits=("train",))
    items = scan_items(records, m)
    assert len(items) == 10 and sum(len(i.views) == 2 for i in items) == 5
    assert all(i.views[0].kernel_class == "sharp" for i in items)
    ds = AdaptDataset(items, [0, 1], SliceSampler("random_k", k=6), augment_prob=1.0)
    pair, single = ds[0], ds[9]
    assert len(pair["views"]) == 2 and pair["aligned"] and (pair["views"][0]["idx"] == pair["views"][1]["idx"]).all()
    assert len(single["views"]) == 1 and single["augment"] == "blur" and pair["views"][0]["hu"].shape == (6, 32, 32)


def test_an_adaptation_run_writes_a_usable_encoder_and_experiment(adapt_project, monkeypatch):
    from ct_model.adapt.trainer import train_adapter
    from ct_model.config import load_encoder_config

    root = adapt_project["root"]
    monkeypatch.chdir(root)
    cfg = parse_adapt_config({
        "name": "tiny_lora", "encoder": str(adapt_project["enc"]), "data_config": str(adapt_project["pre"]),
        "data": {"source": "ctrate", "run": "r1"}, "lora": LORA,
        "aggregator": {"type": "abmil", "hidden_dim": 16, "attn_dim": 8, "branches": "per_label"},
        "slices": {"train_k": 6, "val_k": 6}, "consistency": {"augment_prob": 0.5}, "norm_fit_volumes": 4,
        "optim": {"epochs": 2, "scans_per_step": 2, "accumulation": 2, "warmup_steps": 1, "lr_lora": 1e-2, "lr_head": 1e-2},
        "device": "cpu", "amp_dtype": "float32", "output_dir": str(root / "adapters"),
    })
    logs = []
    res = train_adapter(cfg, log=logs.append, workers=0)
    hist = pd.read_csv(res.dir / "metrics.csv")
    assert hist.epoch.tolist() == [0, 1, 2] and hist.val_pairs.iloc[0] == 5
    assert {"train_bce", "train_emb", "train_pred", "val_pair_cosine", "val_macro_auroc[sharp]"} <= set(hist.columns)
    enc_cfg = load_encoder_config(str(res.encoder_yaml))
    assert enc_cfg.lora["adapter"].endswith("adapter.safetensors")
    adapted = build_encoder(enc_cfg)
    hu = torch.randint(-1000, 500, (2, 32, 32)).float()
    assert not torch.allclose(adapted(hu), _encoder(lora=None, frozen=True)(hu))  # trained: no longer the base model
    exp = yaml.safe_load(res.experiment_yaml.read_text())
    assert exp["encoder"].endswith("encoder.yaml") and exp["data"]["run"] == "r1"
    assert (root / "adapters" / "index.csv").is_file()
    with pytest.raises(Exception, match="already adapted"):
        train_adapter(cfg, log=logs.append, workers=0)
    assert replace(cfg, seed=1).run_dir("abcdef12").name == "seed1"


def test_resume_state_holds_only_what_trains():
    from ct_model.adapt.trainer import EndToEnd, load_trainable_state, trainable_state
    from ct_model.training.config import parse_experiment
    from ct_model.training.trainer import build_model

    def model():
        stub = parse_experiment({"name": "s", "aggregator": {"type": "abmil", "hidden_dim": 8, "attn_dim": 4,
                                                             "branches": "per_label"}, "head": {"type": "linear"}})
        torch.manual_seed(1)
        return EndToEnd(_encoder(), build_model(stub, 192, ["a", "b"]), checkpoint=False)

    a, b = model(), model()
    for p in lora_parameters(a.encoder).values():
        torch.nn.init.normal_(p, std=0.05)
    state = trainable_state(a)
    assert not any(k.startswith("encoder.model.patch_embed") or k.endswith("base.weight") for k in state)
    assert all(k.startswith("classifier.") or "lora_" in k for k in state)
    load_trainable_state(b, state)
    hu = torch.randint(-1000, 500, (3, 32, 32)).float()
    a.eval(), b.eval()
    assert torch.allclose(a.classify([a.embed(hu)]), b.classify([b.embed(hu)]), atol=1e-5)
    with pytest.raises(RuntimeError, match="does not match"):
        load_trainable_state(b, {k: v for k, v in state.items() if "lora_A" not in k})
