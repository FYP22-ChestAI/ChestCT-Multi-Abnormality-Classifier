"""Step 5: assign train / val / test by PATIENT -- once, then frozen.

    python scripts/preprocessing/assign_splits.py

Run after ingest, merge_manifests.py and qc_report.py, and before the first
training run. Only QC-passed volumes count, so a patient whose scans all
failed is not assigned at all, and a volume that failed QC is marked
"excluded" in the manifest.

  * ctrate    -- test = the official valid pool (a rule, not a draw);
                 val = n_val_patients drawn from train-pool patients (seeded);
                 the rest = train.
  * nhrd_local -- no official split: n_val_patients and n_test_patients are
                 drawn (seeded); the rest = train.

The assignment is written to data/splits/<source>.csv and NEVER rewritten:
running this again only assigns patients that are not in the file yet (after a
top-up), leaving every earlier assignment untouched. To start over, delete the
file by hand -- on purpose, never by accident -- and only before any training.

Counts come from configs/preprocessing.yaml (sources.<name>.split); override
them here. With --dry-run nothing is written.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ct_preprocessing.cli import add_config_arg, friendly_errors, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.ingest import state
from ct_preprocessing.ingest.splits import SPLITS, InsufficientPatients, apply_splits, assign_source_splits, read_splits


@friendly_errors
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--source", action="append", default=None, help="a source to split (repeatable; default: every source in the manifest)")
    ap.add_argument("--n-val-patients", type=int, default=None, help="override the validation patient count (one source only)")
    ap.add_argument("--n-test-patients", type=int, default=None, help="override the test patient count (one source only)")
    ap.add_argument("--seed", type=int, default=None, help="override the draw seed (one source only)")
    ap.add_argument("--skip-qc", action="store_true", help="split every ingested volume without reading qc_report.csv")
    ap.add_argument("--dry-run", action="store_true", help="show the result without writing anything")
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg.paths.manifest_path)
    sources = args.source or sorted(manifest["source_name"].unique())
    overrides = (args.n_val_patients, args.n_test_patients, args.seed)
    if any(v is not None for v in overrides) and len(sources) != 1:
        raise SystemExit("--n-val-patients / --n-test-patients / --seed need exactly one --source")

    qc = None
    if not args.skip_qc:
        qc_path = Path(cfg.paths.qc_report_path)
        if not qc_path.exists():
            raise SystemExit(f"{qc_path} not found -- run qc_report.py first (or pass --skip-qc to split every volume)")
        qc = pd.read_csv(qc_path)
    show_settings("using", dict(
        sources=",".join(sources), qc=("skipped" if args.skip_qc else cfg.paths.qc_report_path), dry_run=args.dry_run,
    ))

    frozen: dict[str, dict[str, str]] = {name: read_splits(state.splits_path(cfg.paths, name)) for name in sources}
    for name in sources:
        try:
            report = assign_source_splits(
                manifest, name, cfg.source(name), state.splits_path(cfg.paths, name), qc,
                n_val=args.n_val_patients, n_test=args.n_test_patients, seed=args.seed, dry_run=args.dry_run,
            )
        except (InsufficientPatients, ValueError) as exc:
            raise SystemExit(f"error [{name}]: {exc}") from exc
        print(report.format())
        frozen[name] = report.mapping

    final = apply_splits(manifest, frozen, qc)
    if args.dry_run:
        print("dry run: nothing written")
    else:
        final.to_csv(cfg.paths.manifest_path, index=False)
        print(f"wrote {cfg.paths.manifest_path} (split + qc_passed columns)")
    print("manifest rows by split: " + ", ".join(f"{s}={int((final['split'] == s).sum())}" for s in (*SPLITS, "excluded", "unassigned")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
