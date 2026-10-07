"""Load the model-side YAML files (configs/model/) into typed config objects.

Same rules as ct_preprocessing.config: every tunable value lives in a YAML file, every value is
only a DEFAULT that a script argument can override, and unknown keys or invalid values raise
immediately -- a typo fails loudly instead of silently falling back to a default.

    configs/model/encode.yaml             stage-2 defaults (which volumes, batch sizes, device, store folder)
    configs/model/encoders/<name>.yaml    one backbone each: architecture, weights, input transform, pooling

An encoder's ``fingerprint()`` is a hash of its whole config. It names the embedding store folder,
so embeddings made with different settings (another backbone, input size, pooling, ...) never mix.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

DEFAULT_ENCODE_CONFIG = "configs/model/encode.yaml"
DEFAULT_ENCODERS_DIR = "configs/model/encoders"

_WEIGHT_SOURCES = ("hf_safetensors", "local_safetensors", "timm_pretrained", "none")
_TRANSFORMS = ("clip_zscore", "multi_window")
_RESIZE_MODES = ("bilinear", "bicubic", "nearest")
_POOLINGS = ("cls", "mean_patch", "cls_mean")
_STORE_DTYPES = ("float16", "float32")
_DEVICES = ("auto", "cuda", "cpu")
_AMP_DTYPES = ("auto", "bfloat16", "float16", "float32")
_SPLITS = ("train", "val", "test")


def _check(value, allowed: tuple, name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


@dataclass
class WeightsConfig:
    # hf_safetensors (a safetensors file in a HF repo) | local_safetensors (``filename`` is a local path)
    # | timm_pretrained | none (random init: NOT reproducible, so no in-slice heatmaps from its store)
    source: str = "none"
    hf_repo: str | None = None
    filename: str = "model.safetensors"  # file in the HF repo, or the local path for local_safetensors
    revision: str | None = None  # pin a commit sha: a moving branch could change the embeddings silently
    ignore_prefixes: list[str] = field(default_factory=list)  # keys allowed to be missing/unexpected; all others raise

    def __post_init__(self) -> None:
        _check(self.source, _WEIGHT_SOURCES, "weights.source")
        if self.source == "hf_safetensors" and not self.hf_repo:
            raise ValueError("weights.hf_repo is required when weights.source is hf_safetensors")


@dataclass
class InputConfig:
    """How raw int16 HU slices from the cache become the backbone's input tensor."""

    transform: str = "clip_zscore"
    # clip_zscore (1 channel): (clip(HU, *clip_hu) - mean_hu) / std_hu
    clip_hu: tuple[float, float] | None = None
    mean_hu: float | None = None
    std_hu: float | None = None
    # multi_window (one channel per window): (clip(HU, lo, hi) - lo) / (hi - lo), then optional per-channel norm
    windows: list[tuple[float, float]] | None = None
    channel_mean: list[float] | None = None
    channel_std: list[float] | None = None
    size_hw: tuple[int, int] = (224, 224)  # resized on the device if the cache differs; the cache is never touched
    resize_mode: str = "bilinear"

    def __post_init__(self) -> None:
        _check(self.transform, _TRANSFORMS, "input.transform")
        _check(self.resize_mode, _RESIZE_MODES, "input.resize_mode")
        self.size_hw = tuple(int(v) for v in self.size_hw)
        if len(self.size_hw) != 2 or min(self.size_hw) < 1:
            raise ValueError(f"input.size_hw must be [H, W], got {self.size_hw}")
        if self.transform == "clip_zscore":
            if self.clip_hu is None or self.mean_hu is None or self.std_hu is None:
                raise ValueError("input.transform clip_zscore needs clip_hu, mean_hu and std_hu")
            self.clip_hu = tuple(float(v) for v in self.clip_hu)
            if len(self.clip_hu) != 2 or self.clip_hu[0] >= self.clip_hu[1]:
                raise ValueError(f"input.clip_hu must be [low, high] with low < high, got {self.clip_hu}")
            if self.std_hu <= 0:
                raise ValueError(f"input.std_hu must be > 0, got {self.std_hu}")
        else:
            if not self.windows:
                raise ValueError("input.transform multi_window needs windows: [[lo, hi], ...]")
            self.windows = [tuple(float(v) for v in w) for w in self.windows]
            if any(len(w) != 2 or w[0] >= w[1] for w in self.windows):
                raise ValueError(f"input.windows must be [lo, hi] pairs with lo < hi, got {self.windows}")
            for name in ("channel_mean", "channel_std"):
                value = getattr(self, name)
                if value is not None and len(value) != len(self.windows):
                    raise ValueError(f"input.{name} needs one value per window ({len(self.windows)}), got {value}")

    @property
    def channels(self) -> int:
        return 1 if self.transform == "clip_zscore" else len(self.windows)


@dataclass
class EncoderConfig:
    """One backbone (configs/model/encoders/<name>.yaml)."""

    name: str
    type: str = "timm_vit"  # a builder registered in ct_model.encoders.registry
    arch: str = ""
    model_args: dict = field(default_factory=dict)
    embed_dim: int | None = None  # the per-slice feature size the config promises; checked against the built model
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    input: InputConfig = field(default_factory=InputConfig)
    pooling: str = "cls"
    store_dtype: str = "float16"
    lora: dict | None = None  # phase 2 -- see ct_model.encoders.lora

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("encoder name is required")
        _check(self.pooling, _POOLINGS, "pooling")
        _check(self.store_dtype, _STORE_DTYPES, "store_dtype")

    def fingerprint(self) -> str:
        """A short, stable hash of every setting that affects the embeddings."""
        payload = json.dumps(asdict(self), sort_keys=True, default=list).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def store_name(self) -> str:
        """The embedding store folder name: readable, and unique per config."""
        return f"{self.name}-{self.fingerprint()[:8]}"


@dataclass
class SelectionConfig:
    source: str = "ctrate"
    run: str | None = None
    splits: tuple[str, ...] = _SPLITS
    qc_passed_only: bool = True
    kernel_classes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        self.splits = tuple(self.splits)
        for s in self.splits:
            _check(s, _SPLITS, "selection.splits")
        if self.kernel_classes is not None:
            self.kernel_classes = tuple(self.kernel_classes)


@dataclass
class EncodeConfig:
    embeddings_dir: str = "data/embeddings"
    device: str = "auto"
    amp_dtype: str = "auto"
    slice_batch_size: int = 128
    num_workers: int = 2
    min_free_gb: float = 20.0
    limit: int | None = None

    def __post_init__(self) -> None:
        _check(self.device, _DEVICES, "encode.device")
        _check(self.amp_dtype, _AMP_DTYPES, "encode.amp_dtype")
        if self.slice_batch_size < 1:
            raise ValueError(f"encode.slice_batch_size must be >= 1, got {self.slice_batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"encode.num_workers must be >= 0, got {self.num_workers}")
        if self.limit is not None and self.limit < 1:
            raise ValueError(f"encode.limit must be >= 1 or null, got {self.limit}")


@dataclass
class Stage2Config:
    data_config: str = "configs/preprocessing.yaml"
    encoder: str = "dale_ct_2s"
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)


def _read_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_encoder_config(raw: dict, origin: str = "<dict>") -> EncoderConfig:
    raw = dict(raw or {})
    try:
        weights = WeightsConfig(**(raw.pop("weights", None) or {}))
        input_cfg = InputConfig(**(raw.pop("input", None) or {}))
        return EncoderConfig(weights=weights, input=input_cfg, **raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid encoder config {origin}: {exc}") from exc


def encoder_config_path(name_or_path: str, encoders_dir: str | Path = DEFAULT_ENCODERS_DIR) -> Path:
    """``dale_ct_2s`` -> configs/model/encoders/dale_ct_2s.yaml; a path to a .yaml file is used as given."""
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml"):
        return path
    path = Path(encoders_dir) / f"{name_or_path}.yaml"
    if not path.is_file():
        have = sorted(p.stem for p in Path(encoders_dir).glob("*.yaml")) if Path(encoders_dir).is_dir() else []
        raise FileNotFoundError(f"no encoder config {path}" + (f" (available: {', '.join(have)})" if have else ""))
    return path


def load_encoder_config(name_or_path: str, encoders_dir: str | Path = DEFAULT_ENCODERS_DIR) -> EncoderConfig:
    path = encoder_config_path(name_or_path, encoders_dir)
    return parse_encoder_config(_read_yaml(path), origin=str(path))


def load_stage2_config(path: str | Path = DEFAULT_ENCODE_CONFIG) -> Stage2Config:
    raw = _read_yaml(path)
    try:
        selection = SelectionConfig(**(raw.pop("selection", None) or {}))
        encode = EncodeConfig(**(raw.pop("encode", None) or {}))
        return Stage2Config(selection=selection, encode=encode, **raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid stage-2 config {path}: {exc}") from exc
