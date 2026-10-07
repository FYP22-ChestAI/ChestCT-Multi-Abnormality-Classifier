"""Experiment configs for stages 3 + 4 (configs/model/experiments/<name>.yaml), strict and typed.

Same rules as the other configs: unknown keys and invalid values raise immediately, every value is a
default that a script argument may override, and the resolved config is saved with the results.
``aggregator`` / ``head`` / ``loss`` are ``{type: ..., <params>}`` dicts checked by their registry
(ct_model.registry), so a new component needs no change here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ..data.slices import SliceSampler

DEFAULT_EXPERIMENTS_DIR = "configs/model/experiments"
_MONITORS = ("val_macro_auroc", "val_macro_auprc", "val_loss")
_SPLITS = ("train", "val", "test")


@dataclass
class DataSection:
    source: str = "ctrate"
    run: str | None = None  # null = the only run of the source on disk
    train_kernel_classes: tuple[str, ...] | None = ("sharp",)  # train AND val use these; null = all
    labels: tuple[str, ...] | None = None  # label names without "label_"; null = every label of the manifest
    qc_passed_only: bool = True
    train_slice_sampler: dict = field(default_factory=lambda: {"mode": "all"})  # bag augmentation (SliceSampler)
    preload: bool = True  # bags in RAM (~0.47 MB per volume)

    def __post_init__(self) -> None:
        if self.train_kernel_classes is not None:
            self.train_kernel_classes = tuple(self.train_kernel_classes)
        if self.labels is not None:
            self.labels = tuple(self.labels)
        SliceSampler(**self.train_slice_sampler)  # validates


@dataclass
class FeaturesSection:
    standardize: bool = True  # per-dimension mean / std fitted on TRAIN slices


@dataclass
class OptimSection:
    optimizer: str = "adamw"
    lr: float = 2e-4
    weight_decay: float = 1e-2  # on weight matrices only; never on biases
    batch_size: int = 16  # volumes per step (bags padded + masked)
    max_epochs: int = 100
    warmup_epochs: float = 2.0  # linear warm-up, then cosine decay to 0 at max_epochs
    grad_clip: float | None = 1.0
    patience: int = 15  # epochs without improvement of `monitor` before stopping
    monitor: str = "val_macro_auroc"

    def __post_init__(self) -> None:
        if self.optimizer != "adamw":
            raise ValueError(f"optim.optimizer must be adamw, got {self.optimizer!r}")
        if self.monitor not in _MONITORS:
            raise ValueError(f"optim.monitor must be one of {list(_MONITORS)}, got {self.monitor!r}")
        for name in ("lr", "batch_size", "max_epochs", "patience"):
            if getattr(self, name) <= 0:
                raise ValueError(f"optim.{name} must be > 0, got {getattr(self, name)}")
        if not 0 <= self.warmup_epochs < self.max_epochs:
            raise ValueError("optim.warmup_epochs must be in [0, max_epochs)")


@dataclass
class ExperimentConfig:
    name: str
    data_config: str = "configs/preprocessing.yaml"  # cache_dir / runs_dir
    encode_config: str = "configs/model/encode.yaml"  # embeddings_dir
    encoder: str = "dale_ct_2s"  # which embedding store (configs/model/encoders/<name>.yaml)
    data: DataSection = field(default_factory=DataSection)
    features: FeaturesSection = field(default_factory=FeaturesSection)
    aggregator: dict = field(default_factory=lambda: {"type": "abmil"})
    head: dict = field(default_factory=lambda: {"type": "linear"})
    head_prior_bias: bool = True  # initialise head biases to the train prevalence
    loss: dict = field(default_factory=lambda: {"type": "bce"})
    optim: OptimSection = field(default_factory=OptimSection)
    seed: int = 0
    device: str = "auto"
    output_dir: str = "outputs/experiments"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name is required")
        for part in ("aggregator", "head", "loss"):
            if "type" not in (getattr(self, part) or {}):
                raise ValueError(f"{part} needs a type")
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"device must be auto, cuda or cpu, got {self.device!r}")

    def to_dict(self) -> dict:
        return _plain(asdict(self))


# settings that do not change WHAT is learned: left out of the config hash, so seeds of one config group
NOT_HASHED = ("name", "seed", "device", "output_dir", "data_config", "encode_config")


def _plain(value):
    """Tuples -> lists recursively, so the dict round-trips through YAML / JSON unchanged."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def parse_experiment(raw: dict, origin: str = "<dict>") -> ExperimentConfig:
    raw = dict(raw or {})
    try:
        sections = {
            "data": DataSection(**(raw.pop("data", None) or {})),
            "features": FeaturesSection(**(raw.pop("features", None) or {})),
            "optim": OptimSection(**(raw.pop("optim", None) or {})),
        }
        return ExperimentConfig(**sections, **raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid experiment config {origin}: {exc}") from exc


def experiment_config_path(name_or_path: str, experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR) -> Path:
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml"):
        return path
    path = Path(experiments_dir) / f"{name_or_path}.yaml"
    if not path.is_file():
        have = sorted(p.stem for p in Path(experiments_dir).glob("*.yaml")) if Path(experiments_dir).is_dir() else []
        raise FileNotFoundError(f"no experiment config {path}" + (f" (available: {', '.join(have)})" if have else ""))
    return path


def load_experiment(name_or_path: str, experiments_dir: str | Path = DEFAULT_EXPERIMENTS_DIR) -> ExperimentConfig:
    path = experiment_config_path(name_or_path, experiments_dir)
    with open(path, encoding="utf-8") as f:
        return parse_experiment(yaml.safe_load(f) or {}, origin=str(path))


def override(cfg: ExperimentConfig, **changes) -> ExperimentConfig:
    """A copy with dotted-key overrides (``{"optim.lr": 1e-4, "loss": {"type": "asl"}}``); None = keep."""
    raw = cfg.to_dict()
    for key, value in changes.items():
        if value is None:
            continue
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"no setting {key!r} in the experiment config")
        node[leaf] = value
    return parse_experiment(raw, origin="overrides")
