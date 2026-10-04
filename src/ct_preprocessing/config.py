"""Load configs/preprocessing.yaml into typed config objects used across scripts.

Keeping every tunable number (spacing, size, thresholds, paths, ingest and
split defaults) in one YAML file -- not scattered through code -- means the
whole team agrees on one set of settings. Every value is only a DEFAULT: each
script takes a command-line argument that overrides it for one run.

Unknown keys and invalid values raise immediately, so a typo in the YAML
fails loudly instead of silently falling back to a default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .preprocess_config import PreprocessConfig
from .quality import QCThresholds

DEFAULT_CONFIG_PATH = "configs/preprocessing.yaml"

_FORMATS = ("nifti", "dicom", "auto")
_BUILDERS = ("ctrate", "folder")
_PATIENT_ID_SOURCES = ("auto", "path", "dicom_tag")
_POOLS = ("one_per_scan", "all")


def _check(value, allowed: tuple, name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"  # parent of every source's scratch folder (see SourceConfig.raw_dir)
    cache_dir: str = "data/cache"  # the permanent output: one .npy + .meta.json per volume
    manifest_path: str = "data/manifest.csv"
    qc_report_path: str = "data/qc_report.csv"
    montage_dir: str = "data/qc_montages"
    preprocessing_manifest_path: str = "data/preprocessing_manifest.json"
    metadata_dir: str = "data/metadata"  # CT-RATE metadata CSVs
    worklist_dir: str = "data/worklists"  # make_worklist.py output (CT-RATE)
    chunk_manifest_dir: str = "data/manifests"  # one small manifest per ingested chunk
    state_dir: str = "data/ingest_state"  # .done / .failed markers, so ingest can resume
    splits_dir: str = "data/splits"  # frozen patient -> split files


@dataclass
class IngestConfig:
    """Defaults for scripts/preprocessing/{make_worklist,ingest}.py, per source."""

    # --- all sources
    min_free_gb: float = 100.0  # stop cleanly (never crash the server) when free disk drops below this
    workers: int = 4  # parallel preprocessing processes
    est_mb_per_volume: float = 22.0  # only for the projected-cache-size estimate; replace with the measured value
    # --- CT-RATE: the worklist and how it is fetched
    chunk_size: int = 40  # volumes fetched, preprocessed and cleaned up together
    train_pool: str = "one_per_scan"  # "one_per_scan" keeps one reconstruction per scan; "all" keeps both
    test_pool: str = "all"  # the whole valid_fixed pool by default (comparable with published CT-RATE numbers)
    max_train_patients: int | None = None  # cap the train pool if the calibration run says the cache will not fit
    max_test_patients: int | None = None  # cap the test pool (only for small pilot runs; the default keeps it whole)
    max_combined_gb: float | None = None  # skip TRAIN volumes needing more resample memory than this (metadata only)
    seed: int = 0  # worklist shuffle seed
    hf_repo: str = "ibrahimhamamci/CT-RATE"
    fetch_workers: int = 4  # parallel downloads inside one chunk
    # --- archive sources (NHRD): where the uploaded .zip/.tar archives are
    drive_remote: str | None = None  # an rclone remote like "gdrive:nhrd_raw", or a plain folder path

    def __post_init__(self) -> None:
        _check(self.train_pool, _POOLS, "ingest.train_pool")
        _check(self.test_pool, _POOLS, "ingest.test_pool")
        if self.chunk_size < 1:
            raise ValueError(f"ingest.chunk_size must be >= 1, got {self.chunk_size}")
        if self.workers < 1 or self.fetch_workers < 1:
            raise ValueError("ingest.workers and ingest.fetch_workers must be >= 1")
        if self.min_free_gb < 0:
            raise ValueError(f"ingest.min_free_gb must be >= 0, got {self.min_free_gb}")


@dataclass
class SplitConfig:
    """Defaults for scripts/preprocessing/assign_splits.py, per source.

    Splits are always by PATIENT, decided once after ingest and QC, then
    frozen. CT-RATE's test set is its own official valid pool (not drawn
    here), so only n_val_patients applies to it.
    """

    n_val_patients: int | None = None
    n_test_patients: int | None = None
    seed: int = 0


@dataclass
class SourceConfig:
    """One named data source (CT-RATE, the local NHRD hospital data, ...)."""

    format: str = "nifti"  # "nifti", "dicom", or "auto" (detect from the files found)
    raw_dir: str = "data/raw"  # SCRATCH: ingest fills it per chunk and empties it again. Never point it at real data.
    manifest_builder: str = "ctrate"  # "ctrate" (parsed ids + metadata CSV) or "folder" (discover scans in a tree)
    # How scans are grouped into patients. "auto" decides from the DICOM
    # PatientID tag's actual presence/uniqueness (and sticks with the first
    # decision for the whole source); "path" uses the first
    # `patient_path_depth` folder levels; "dicom_tag" forces the tag.
    patient_id_source: str = "auto"
    patient_path_depth: int = 1
    # Per-source overrides of the global `qc:` thresholds, e.g. {"min_slices": 60}.
    qc: dict = field(default_factory=dict)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    split: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self) -> None:
        _check(self.format, _FORMATS, "format")
        _check(self.manifest_builder, _BUILDERS, "manifest_builder")
        _check(self.patient_id_source, _PATIENT_ID_SOURCES, "patient_id_source")


@dataclass
class DataConfig:
    paths: PathsConfig
    preprocess: PreprocessConfig
    qc: QCThresholds
    sources: dict[str, SourceConfig] = field(default_factory=dict)

    def source(self, name: str) -> SourceConfig:
        if name not in self.sources:
            raise KeyError(f"unknown source {name!r}; configured sources: {sorted(self.sources)}")
        return self.sources[name]

    def qc_for(self, source_name: str | None) -> QCThresholds:
        """The global QC thresholds, with this source's overrides applied."""
        source = self.sources.get(source_name) if source_name else None
        if source is None or not source.qc:
            return self.qc
        return QCThresholds(**{**asdict(self.qc), **source.qc})


def _parse_source(name: str, raw: dict) -> SourceConfig:
    raw = dict(raw or {})
    try:
        ingest = IngestConfig(**(raw.pop("ingest", None) or {}))
        split = SplitConfig(**(raw.pop("split", None) or {}))
        return SourceConfig(ingest=ingest, split=split, **raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid config for source {name!r}: {exc}") from exc


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> DataConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    pre_raw = dict(raw.get("preprocess", {}))
    if "target_spacing_zyx" in pre_raw:
        pre_raw["target_spacing_zyx"] = tuple(pre_raw["target_spacing_zyx"])
    if "target_size_hw" in pre_raw:
        pre_raw["target_size_hw"] = tuple(pre_raw["target_size_hw"])

    sources_raw = raw.get("sources", {}) or {}
    return DataConfig(
        paths=PathsConfig(**raw.get("paths", {})),
        preprocess=PreprocessConfig(**pre_raw),
        qc=QCThresholds(**raw.get("qc", {})),
        sources={name: _parse_source(name, cfg) for name, cfg in sources_raw.items()},
    )
