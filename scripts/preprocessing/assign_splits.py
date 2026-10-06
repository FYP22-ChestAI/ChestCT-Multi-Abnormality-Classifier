"""Step 5: assign train / val / test by PATIENT -- once, then frozen.

    python scripts/preprocessing/assign_splits.py --source ctrate --run train-sharp

Run after ingest, merge_manifests.py and qc_report.py, and before the first training run. Only
QC-passed volumes count, so a patient whose scans all failed is not assigned at all, and a volume
that failed QC is marked "excluded" in the manifest.

  * ctrate    -- test = the official valid pool (a rule, not a draw);
                 val = n_val_patients drawn from train-pool patients (seeded);
                 the rest = train.
  * nhrd_local -- no official split: n_val_patients and n_test_patients are
                 drawn (seeded); the rest = train.

There is ONE split file per source, data/splits/<source>.csv, shared by every run: a patient has the
same split in a sharp run and in a soft run, which is what keeps runs from leaking into each other.
It is NEVER rewritten: running this again (for another run) only assigns patients that are not in
the file yet, leaving every earlier assignment untouched. To start over, delete the file by hand --
on purpose, never by accident -- and only before any training. A copy limited to the run's patients is
written to the run folder (splits.csv), so the folder is self-contained.

Counts come from configs/preprocessing.yaml (sources.<name>.split); override them here. With
--dry-run nothing is written.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ct_preprocessing.cli import add_config_arg, friendly_errors, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.ingest import state
from ct_preprocessing.ingest.splits import SPLITS, InsufficientPatients, apply_splits, assign_source_splits, write_splits
from ct_preprocessing.runs import resolve_run


@friendly_errors
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--source", required=True, help="the source of the run to split (ctrate, nhrd_local, ...)")
    ap.add_argument("--run", default=None, help="the run to split (default: the only one)")
    ap.add_argument("--n-val-patients", type=int, default=None, help="override the validation patient count")
    ap.add_argument("--n-test-patients", type=int, default=None, help="override the test patient count")
    ap.add_argument("--seed", type=int, default=None, help="override the draw seed")
    ap.add_argument("--skip-qc", action="store_true", help="split every ingested volume without reading qc_report.csv")
    ap.add_argument("--dry-run", action="store_true", help="show the result without writing anything")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = resolve_run(cfg.paths, args.source, args.run)
    if not run.manifest_path.exists():
        raise FileNotFoundError(f"{run.manifest_path} not found -- run merge_manifests.py --source {args.source} --run {run.name} first")
    manifest = pd.read_csv(run.manifest_path)

    qc = None
    if not args.skip_qc:
        if not run.qc_report_path.exists():
            raise FileNotFoundError(
                f"{run.qc_report_path} not found -- run qc_report.py --source {args.source} --run {run.name} first "
                "(or pass --skip-qc to split every volume)"
            )
        qc = pd.read_csv(run.qc_report_path)
    show_settings("using", dict(
        source=args.source, run=run.name, qc=("skipped" if args.skip_qc else run.qc_report_path), dry_run=args.dry_run,
    ))

    splits_file = state.splits_path(cfg.paths, args.source)
    try:
        report = assign_source_splits(
            manifest, args.source, cfg.source(args.source), splits_file, qc,
            n_val=args.n_val_patients, n_test=args.n_test_patients, seed=args.seed, dry_run=args.dry_run,
        )
    except (InsufficientPatients, ValueError) as exc:
        raise SystemExit(f"error [{args.source}]: {exc}") from exc
    print(report.format())

    final = apply_splits(manifest, {args.source: report.mapping}, qc)
    if args.dry_run:
        print("dry run: nothing written")
    else:
        final.to_csv(run.manifest_path, index=False)
        in_run = {p: s for p, s in report.mapping.items() if p in set(final["patient_id"])}
        write_splits(run.splits_copy_path, in_run)
        print(f"wrote {run.manifest_path} (split + qc_passed columns) and {run.splits_copy_path}")
    print("manifest rows by split: " + ", ".join(f"{s}={int((final['split'] == s).sum())}" for s in (*SPLITS, "excluded", "unassigned")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
