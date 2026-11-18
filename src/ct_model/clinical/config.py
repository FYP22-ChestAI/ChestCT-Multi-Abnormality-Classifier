"""configs/model/clinical.yaml -> a typed ClinicalConfig. Unknown keys and invalid values raise, like
the other model configs (ct_model.config), so a typo never silently falls back to a default."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

DEFAULT_CLINICAL_CONFIG = "configs/model/clinical.yaml"
_REPO = Path(__file__).resolve().parents[3]
_CALIBRATIONS = ("none", "temperature", "platt")
_GROUPINGS = ("all", "kernel_class", "manufacturer")


def _fraction(value: float, name: str) -> float:
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}")
    return float(value)


@dataclass
class OperatingPoints:
    report_specificity: float = 0.95
    ruleout_sensitivity: float = 0.90
    calibration: str = "platt"
    calibration_by_kernel: bool = False
    min_calibration_volumes: int = 100

    def __post_init__(self) -> None:
        _fraction(self.report_specificity, "operating_points.report_specificity")
        _fraction(self.ruleout_sensitivity, "operating_points.ruleout_sensitivity")
        if self.calibration not in _CALIBRATIONS:
            raise ValueError(f"operating_points.calibration must be one of {list(_CALIBRATIONS)}, got {self.calibration!r}")


@dataclass
class Gate:
    auroc_ci_low: float = 0.80
    report_sensitivity: float = 0.60
    ece: float = 0.05
    paired_flip_rate: float = 0.10
    kernel_sensitivity_gap: float = 0.10


@dataclass
class ClinicalConfig:
    operating_points: OperatingPoints = field(default_factory=OperatingPoints)
    fixed_specificities: tuple[float, ...] = (0.90, 0.95)
    net_benefit_thresholds: tuple[float, ...] = (0.05, 0.10, 0.20)
    calibration_bins: int = 10
    bootstrap: int = 1000
    ci_groups: tuple[str, ...] = ("all", "kernel_class")
    min_group: int = 100
    gate: Gate = field(default_factory=Gate)

    def __post_init__(self) -> None:
        self.fixed_specificities = tuple(_fraction(s, "fixed_specificities") for s in self.fixed_specificities)
        self.net_benefit_thresholds = tuple(_fraction(t, "net_benefit_thresholds") for t in self.net_benefit_thresholds)
        self.ci_groups = tuple(self.ci_groups)
        bad = set(self.ci_groups) - set(_GROUPINGS)
        if bad:
            raise ValueError(f"ci_groups must be among {list(_GROUPINGS)}, got {sorted(bad)}")
        if self.calibration_bins < 2:
            raise ValueError(f"calibration_bins must be >= 2, got {self.calibration_bins}")

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Short hash of every setting: two reports with the same fingerprint used the same yardstick."""
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _section(cls, raw: dict | None, name: str):
    raw = raw or {}
    known = set(cls.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown key(s) in {name}: {sorted(unknown)} (known: {sorted(known)})")
    return cls(**raw)


def parse_clinical_config(raw: dict | None, origin: str = "<dict>") -> ClinicalConfig:
    raw = dict(raw or {})
    try:
        raw["operating_points"] = _section(OperatingPoints, raw.get("operating_points"), "operating_points")
        raw["gate"] = _section(Gate, raw.get("gate"), "gate")
        return _section(ClinicalConfig, raw, "the clinical config")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{origin}: {exc}") from exc


def load_clinical_config(path: str | Path = DEFAULT_CLINICAL_CONFIG) -> ClinicalConfig:
    """Relative paths are tried from the working directory, then from the repository root."""
    path = Path(path)
    if not path.is_absolute() and not path.is_file() and (_REPO / path).is_file():
        path = _REPO / path
    with open(path, encoding="utf-8") as f:
        return parse_clinical_config(yaml.safe_load(f), origin=str(path))
