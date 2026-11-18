"""Clinical evaluation of an evaluated run -- or of a seed ensemble -- from its saved predictions.

    python scripts/model/clinical_report.py --experiment-dir outputs/experiments/abmil_dale2s/1a2b3c4d/seed0
    python scripts/model/clinical_report.py --experiment-dir outputs/experiments/abmil_dale2s/1a2b3c4d --per-seed

A run folder (``.../seed<k>``) gets its own report; a config folder (``.../<hash8>``) gets the report of
the seed ensemble (probabilities averaged over every seed that has this evaluation), and with
--per-seed every seed's report as well. evaluate_mil.py already writes the run's own report, so this
script is for ensembles, for re-scoring with changed clinical settings, and for older evaluations.

Operating points and calibration are chosen on VAL (predictions_val.csv) and applied unchanged to the
evaluated split. Settings: configs/model/clinical.yaml. Writes <folder>/clinical/<source>__<run>__<split>/
(or <config folder>/ensemble/clinical/...) and adds a row to outputs/experiments/clinical.csv.
"""
from __future__ import annotations

import argparse

from ct_preprocessing.cli import friendly_errors, show_settings

from ct_model.utils.log import stream_logs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-dir", required=True,
                    help="a run (outputs/experiments/<name>/<hash>/seed<k>) or a config folder (<name>/<hash>: seed ensemble)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"], help="the evaluation to report")
    ap.add_argument("--source", help="default: the experiment's source")
    ap.add_argument("--run", help="default: the run the experiment was trained on")
    ap.add_argument("--clinical-config", default="configs/model/clinical.yaml", help="clinical settings (operating points, gate)")
    ap.add_argument("--per-seed", action="store_true", help="config folder: also write every seed's own report")
    ap.add_argument("--bootstrap", type=int, help="patient resamples for the AUROC CIs (default: from --clinical-config; 0 = none)")
    return ap


@friendly_errors
def main() -> int:
    stream_logs()  # progress lines reach | tee / log files as they happen
    args = build_parser().parse_args()
    show_settings("clinical_report", {k: v for k, v in vars(args).items()})

    from ct_model.clinical import clinical_report, load_clinical_config

    cfg = load_clinical_config(args.clinical_config)
    if args.bootstrap is not None:
        cfg.bootstrap = args.bootstrap
    results = clinical_report(args.experiment_dir, split=args.split, source=args.source, run=args.run,
                              config=cfg, per_seed=args.per_seed)
    labels = results[-1].scorecard["labels"]
    for status, names in labels.items():
        print(f"{status:>14}: {', '.join(names) if names else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
