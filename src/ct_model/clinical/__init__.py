"""Clinical evaluation: operating points chosen on val, then what a reader of the output would get
(false labels and missed findings per scan, decisions that change with the reconstruction kernel,
calibration, a per-label gate). Reads only saved predictions, so every phase is scored the same way.

    from ct_model.clinical import clinical_report
    clinical_report("outputs/experiments/<name>/<hash8>/seed0")        # one run
    clinical_report("outputs/experiments/<name>/<hash8>", per_seed=True)  # seed ensemble (+ each seed)
"""
from .config import ClinicalConfig, load_clinical_config
from .report import clinical_report

__all__ = ["ClinicalConfig", "clinical_report", "load_clinical_config"]
