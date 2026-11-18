"""configs/model/adapt/<name>.yaml -> a typed AdaptConfig (phase 2: LoRA adaptation of the encoder).

Like the other configs: unknown keys and invalid values raise. The CONFIG HASH (everything except name, seed,
device and paths, plus the run manifest) names the output folder, outputs/adapters/<name>/<hash8>/seed<k>/.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

DEFAULT_ADAPT_DIR = "configs/model/adapt"
NOT_HASHED = ("name", "seed", "device", "output_dir", "data_config", "encode_config")
_MONITORS = ("val_macro_auroc", "val_loss")


@dataclass
class AdaptData:
    source: str = "ctrate"
    run: str | None = None                       # "a+b" joins runs (ct_model.data.volumes.load_run_manifest)
    kernel_classes: tuple[str, ...] | None = None  # null = every kernel (sharp and soft)
    labels: tuple[str, ...] | None = None        # null = the labels of init_from (or every label of the manifest)
    qc_passed_only: bool = True

    def __post_init__(self) -> None:
        if self.kernel_classes is not None:
            self.kernel_classes = tuple(self.kernel_classes)
        if self.labels is not None:
            self.labels = tuple(self.labels)


@dataclass
class Slices:
    train_k: int = 48          # slices per volume with gradients (random_k: one per equal segment along the body axis)
    val_k: int = 48            # uniform_k for validation
    max_offset: int = 2        # pair alignment: search the z offset between two reconstructions in [-m, m]

    def __post_init__(self) -> None:
        if self.train_k < 1 or self.val_k < 1:
            raise ValueError("slices.train_k and slices.val_k must be >= 1")


@dataclass
class Consistency:
    pairs: bool = True               # a scan with both kernels: its sharp and soft volumes in the same step
    embedding_weight: float = 1.0    # 1 - cosine between the aligned slices of the two views
    prediction_weight: float = 0.5   # symmetric KL between the two views' label probabilities
    augment_prob: float = 0.25       # a one-kernel scan gets a kernel-simulating second view with this probability
    blur_sigma: tuple[float, float] = (0.5, 1.2)     # sharp -> "soft": Gaussian blur, pixels
    sharpen_amount: tuple[float, float] = (0.5, 1.5)  # soft -> "sharp": unsharp mask amount (sigma 1 px)

    def __post_init__(self) -> None:
        self.blur_sigma, self.sharpen_amount = tuple(self.blur_sigma), tuple(self.sharpen_amount)
        if not 0.0 <= self.augment_prob <= 1.0:
            raise ValueError(f"consistency.augment_prob must be in [0, 1], got {self.augment_prob}")


@dataclass
class AdaptOptim:
    lr_lora: float = 1.0e-4
    lr_head: float = 2.0e-5          # aggregator + head (initialised from init_from)
    weight_decay: float = 1.0e-2
    scans_per_step: int = 2          # scans per forward/backward (each 1 or 2 volumes)
    accumulation: int = 8            # optimiser step every this many forward/backward passes
    epochs: int = 6
    warmup_steps: int = 100          # optimiser steps of linear warm-up, then cosine to 0
    grad_clip: float = 1.0
    scans_per_epoch: int | None = None  # null = every train scan once per epoch
    monitor: str = "val_macro_auroc"
    checkpoint: bool = True          # gradient checkpointing in the adapted blocks

    def __post_init__(self) -> None:
        if self.monitor not in _MONITORS:
            raise ValueError(f"optim.monitor must be one of {list(_MONITORS)}, got {self.monitor!r}")


@dataclass
class AdaptConfig:
    name: str
    encoder: str = "dale_ct_2s"                    # the base encoder (configs/model/encoders/<name>.yaml)
    init_from: str | None = None                   # a finished phase-1 run: FeatureNorm, aggregator and head
    data_config: str = "configs/preprocessing.yaml"
    encode_config: str = "configs/model/encode.yaml"
    data: AdaptData = field(default_factory=AdaptData)
    lora: dict = field(default_factory=lambda: {"r": 8, "alpha": 16, "dropout": 0.05,
                                                "target_modules": ["qkv", "proj"], "last_n_blocks": 12})
    slices: Slices = field(default_factory=Slices)
    consistency: Consistency = field(default_factory=Consistency)
    optim: AdaptOptim = field(default_factory=AdaptOptim)
    aggregator: dict = field(default_factory=lambda: {"type": "abmil", "branches": "per_label"})  # only without init_from
    head: dict = field(default_factory=lambda: {"type": "linear"})                              # only without init_from
    norm_fit_volumes: int = 64                     # only without init_from: train volumes to fit FeatureNorm on
    seed: int = 0
    device: str = "auto"
    amp_dtype: str = "bfloat16"
    output_dir: str = "outputs/adapters"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("adapt config name is required")
        if "adapter" in (self.lora or {}):
            raise ValueError("lora.adapter is written by the trainer -- leave it out of the adapt config")

    def to_dict(self) -> dict:
        return _plain(asdict(self))

    def config_hash(self, manifest_sha256: str) -> str:
        settings = self.to_dict()
        for key in NOT_HASHED:
            settings.pop(key, None)
        payload = {"config": settings, "manifest_sha256": manifest_sha256}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def run_dir(self, cfg_hash: str) -> Path:
        return Path(self.output_dir) / self.name / cfg_hash[:8] / f"seed{self.seed}"


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def parse_adapt_config(raw: dict, origin: str = "<dict>") -> AdaptConfig:
    raw = dict(raw or {})
    try:
        sections = {
            "data": AdaptData(**(raw.pop("data", None) or {})),
            "slices": Slices(**(raw.pop("slices", None) or {})),
            "consistency": Consistency(**(raw.pop("consistency", None) or {})),
            "optim": AdaptOptim(**(raw.pop("optim", None) or {})),
        }
        return AdaptConfig(**sections, **raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid adapt config {origin}: {exc}") from exc


def adapt_config_path(name_or_path: str, adapt_dir: str | Path = DEFAULT_ADAPT_DIR) -> Path:
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml"):
        return path
    path = Path(adapt_dir) / f"{name_or_path}.yaml"
    if not path.is_file():
        have = sorted(p.stem for p in Path(adapt_dir).glob("*.yaml")) if Path(adapt_dir).is_dir() else []
        raise FileNotFoundError(f"no adapt config {path}" + (f" (available: {', '.join(have)})" if have else ""))
    return path


def load_adapt_config(name_or_path: str, adapt_dir: str | Path = DEFAULT_ADAPT_DIR) -> AdaptConfig:
    path = adapt_config_path(name_or_path, adapt_dir)
    with open(path, encoding="utf-8") as f:
        return parse_adapt_config(yaml.safe_load(f) or {}, origin=str(path))
