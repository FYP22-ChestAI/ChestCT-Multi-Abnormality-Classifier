"""Load configs/preprocessing.yaml into typed config objects used across scripts.

Keeping every tunable number (spacing, size, windows, thresholds, paths) in
one YAML file -- not scattered through code -- means the whole team agrees on
one set of settings, and preprocessing_manifest.json can record exactly what
was used (see docs/preprocessing/data_contract.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .preprocess_config import PreprocessConfig
from .quality import QCThresholds
from .windows import DEFAULT_WINDOWS


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    cache_dir: str = "data/cache"
    manifest_path: str = "data/manifest.csv"
    qc_report_path: str = "data/qc_report.csv"
    montage_dir: str = "data/qc_montages"
    preprocessing_manifest_path: str = "data/preprocessing_manifest.json"


@dataclass
class SourceConfig:
    """One named data source (CT-RATE train, CT-RATE valid, the local NHRD
    hospital data, ...). Each source declares its own file format and where
    its raw files live -- a plain filesystem path works identically whether
    that's a folder on a laptop or a mounted university server; the only
    thing that's ever special about CT-RATE is the one-time download that
    populates it (see scripts/download_subset.py and docs/preprocessing/data_contract.md).
    """

    format: str = "nifti"  # "nifti", "dicom", or "auto" (detect from the files found)
    raw_dir: str = "data/raw"  # default location; overridable at run time with --source-root / --raw-dir
    manifest_builder: str = "ctrate"  # a key in ct_preprocessing.manifest.MANIFEST_BUILDERS
    # How scans are grouped into patients for patient-level splitting. "auto"
    # (the default) decides for itself, per source, from the DICOM PatientID
    # tag's actual presence/uniqueness (see build_manifest_folder); "path"
    # forces the first `patient_path_depth` folder levels (NHRD: depth 1,
    # since every top folder is one patient and 'P00001' repeats across
    # them); "dicom_tag" forces the tag even if the auto-check would not
    # have picked it.
    patient_id_source: str = "auto"
    patient_path_depth: int = 1
    # Per-source overrides of the global `qc:` thresholds (a local protocol may
    # legitimately differ from CT-RATE), e.g. {"min_slices": 60}.
    qc: dict = field(default_factory=dict)
    # Acquisition amounts -- how many patients scripts/download_subset.py and
    # scripts/build_manifest.py use when no --n-train/--n-val/--n-test/--seed
    # is given on the command line. None means "this value is required on
    # the command line for this source" (no sensible one-size-fits-all
    # default exists yet); set real numbers here once you know your usual
    # pilot/production sizes, so the scripts run with zero arguments.
    n_train: int | None = None
    n_val: int | None = None
    n_test: int | None = None
    seed: int = 0
    max_combined_gb: float | None = None  # CT-RATE only; skips volumes needing too much resample memory


@dataclass
class DataConfig:
    paths: PathsConfig
    preprocess: PreprocessConfig
    qc: QCThresholds
    windows: dict[str, tuple[float, float]]
    sources: dict[str, SourceConfig] = field(default_factory=dict)

    def qc_for(self, source_name: str | None) -> QCThresholds:
        """The global QC thresholds, with this source's overrides applied."""
        source = self.sources.get(source_name) if source_name else None
        if source is None or not source.qc:
            return self.qc
        return QCThresholds(**{**asdict(self.qc), **source.qc})


def load_config(path: str | Path = "configs/preprocessing.yaml") -> DataConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    pre_raw = dict(raw.get("preprocess", {}))
    if "target_spacing_zyx" in pre_raw:
        pre_raw["target_spacing_zyx"] = tuple(pre_raw["target_spacing_zyx"])
    if "target_size_hw" in pre_raw:
        pre_raw["target_size_hw"] = tuple(pre_raw["target_size_hw"])

    windows_raw = raw.get("windows", DEFAULT_WINDOWS)
    windows = {k: tuple(v) for k, v in windows_raw.items()}

    sources_raw = raw.get("sources", {}) or {}
    sources = {name: SourceConfig(**cfg) for name, cfg in sources_raw.items()}

    return DataConfig(
        paths=PathsConfig(**raw.get("paths", {})),
        preprocess=PreprocessConfig(**pre_raw),
        qc=QCThresholds(**raw.get("qc", {})),
        windows=windows,
        sources=sources,
    )
