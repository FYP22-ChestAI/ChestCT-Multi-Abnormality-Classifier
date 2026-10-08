"""Step 1 (CT-RATE only): plan a RUN -- exactly which volumes to ingest.

    python scripts/preprocessing/make_worklist.py                           # run "train-sharp"
    python scripts/preprocessing/make_worklist.py --train-kernel soft       # run "train-soft"
    python scripts/preprocessing/make_worklist.py --train-kernel soft \\
        --from-run train-sharp --train-patients 800 --val-patients 200      # a subset of an existing run's patients

Downloads CT-RATE's small metadata CSVs (~16 MB) and label CSVs (~3 MB), no images, then creates
a new run folder data/runs/<source>/<name>/ with its worklist.csv. A run is NEVER overwritten or
deleted: a different plan is a different run, and the cache of preprocessed volumes is shared, so a
new run only downloads what no earlier run cached.

  * the test pool (CT-RATE's official valid_fixed set) is always taken whole, with every
    reconstruction -- both kernels -- so a model can be tested on sharp and on soft;
  * the train pool is taken one reconstruction per scan, of --train-kernel (sharp = lung kernel,
    soft = mediastinal); a scan without that kernel is skipped. Which pairs are sharp or soft comes
    from configs/kernel_classes.csv (see make_kernel_table.py);
  * --from-run builds the run from patients the PARENT run has already split: --train-patients drawn
    from its train split and --val-patients from its val split. Each keeps the split it has, so there
    is no leakage and the train / val mix is the one you ask for. Run assign_splits.py on the parent first;
  * --survey N makes a small run for the kernel survey: N scans (both reconstructions of each) per
    kernel pair, to check the sharp / soft table by measurement (then kernel_survey.py).

This is a work list, NOT a split. Every default comes from configs/preprocessing.yaml
(sources.ctrate.ingest); override any of them here. Needs your own Hugging Face login:
`huggingface-cli login`, after accepting the CT-RATE terms on the dataset page.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from ct_preprocessing.cli import add_config_arg, friendly_errors, pick, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.ingest import state
from ct_preprocessing.ingest.ctrate import download_metadata, load_metadata
from ct_preprocessing.ingest.labels import download_ctrate_labels
from ct_preprocessing.ingest.sources import get_hf_download
from ct_preprocessing.ingest.splits import read_splits
from ct_preprocessing.ingest.worklist import (
    build_survey_worklist, build_worklist, format_summary, patients_with_kernel, read_worklist,
    select_child_patients, write_worklist,
)
from ct_preprocessing.kernels import load_kernel_table
from ct_preprocessing.manifest import CTRATE_BUILDER
from ct_preprocessing.preprocess import config_fingerprint, is_cache_fresh
from ct_preprocessing.runs import Run, auto_run_name, check_run_name, get_run, new_run, write_run_info

SURVEY_RUN = "kernel-survey"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--source", default="ctrate", help="a CT-RATE-type source in the config (default: ctrate)")
    ap.add_argument("--name", default=None, help="run name (default: train-<kernel>, or kernel-survey)")
    ap.add_argument("--train-kernel", choices=["sharp", "soft"], default=None, help="which reconstruction of each train scan to take")
    ap.add_argument("--chunk-size", type=int, default=None, help="volumes per chunk")
    ap.add_argument("--max-train-patients", type=int, default=None, help="cap the train pool to this many patients")
    ap.add_argument("--max-test-patients", type=int, default=None, help="cap the test pool (for small pilot runs only)")
    ap.add_argument("--max-combined-gb", type=float, default=None, help="skip TRAIN volumes needing more resample memory than this")
    ap.add_argument("--seed", type=int, default=None, help="patient shuffle seed")
    ap.add_argument("--from-run", default=None, help="build from the patients an existing run already split")
    ap.add_argument("--train-patients", type=int, default=None, help="with --from-run: patients drawn from the parent's train split")
    ap.add_argument("--val-patients", type=int, default=None, help="with --from-run: patients drawn from the parent's val split")
    ap.add_argument("--survey", type=int, default=None, metavar="N", help="a kernel-survey run: N scans per (manufacturer, kernel) pair")
    ap.add_argument("--refresh-metadata", action="store_true", help="download the metadata and label CSVs again")
    return ap


@friendly_errors
def main() -> int:
    args = build_parser().parse_args()

    cfg = load_config(args.config)
    source_cfg = cfg.source(args.source)
    if source_cfg.manifest_builder != CTRATE_BUILDER:
        raise SystemExit(f"{args.source!r} is not a CT-RATE-type source -- only those have a worklist")
    ing = source_cfg.ingest

    kernel = pick(args.train_kernel, ing.train_kernel)
    name = check_run_name(args.name or (SURVEY_RUN if args.survey else auto_run_name(kernel)))
    if args.survey and (args.from_run or args.max_train_patients):
        raise SystemExit("--survey cannot be combined with --from-run or --max-train-patients")
    if args.from_run:
        if args.train_patients is None or args.val_patients is None:
            raise SystemExit("--from-run needs both --train-patients and --val-patients")
        if args.max_train_patients is not None:
            raise SystemExit("--from-run chooses the patients itself: use --train-patients / --val-patients, not --max-train-patients")
    elif args.train_patients is not None or args.val_patients is not None:
        raise SystemExit("--train-patients / --val-patients only make sense with --from-run")

    settings = dict(
        source=args.source, run=name, train_kernel=kernel, chunk_size=pick(args.chunk_size, ing.chunk_size),
        max_train_patients=pick(args.max_train_patients, ing.max_train_patients),
        max_test_patients=pick(args.max_test_patients, ing.max_test_patients),
        max_combined_gb=pick(args.max_combined_gb, ing.max_combined_gb), seed=pick(args.seed, ing.seed),
        from_run=args.from_run, survey=args.survey,
    )
    show_settings("using", settings)

    target = Run(cfg.paths, args.source, name)
    if target.dir.exists():
        raise SystemExit(f"error: run {name!r} already exists ({target.dir}) -- runs are never overwritten. Use it as is "
                         f"(--run {name} in the other scripts) or choose another name with --name.")

    download_metadata(cfg.paths, ing.hf_repo, get_hf_download(), refresh=args.refresh_metadata)
    for warning in download_ctrate_labels(cfg.paths, ing.hf_repo, get_hf_download(), refresh=args.refresh_metadata):
        print(f"warning: {warning}")
    meta = load_metadata(cfg.paths)
    table = load_kernel_table(cfg.paths.kernel_table)

    parent = None
    if args.survey:
        worklist, summary = build_survey_worklist(
            meta["train"], kernel_table=table, pairs_per_kernel=args.survey, chunk_size=settings["chunk_size"], seed=settings["seed"],
        )
    else:
        child_patients = None
        if args.from_run:
            parent = get_run(cfg.paths, args.source, args.from_run)
            parent_patients = set(read_worklist(parent.worklist_path).query("source_split == 'train'")["patient_id"])
            splits = {p: s for p, s in read_splits(state.splits_path(cfg.paths, args.source)).items() if p in parent_patients}
            if not splits:
                raise SystemExit(f"run {parent.name!r} has no frozen split yet -- run merge_manifests.py, qc_report.py and "
                                 f"assign_splits.py with --run {parent.name} first")
            child_patients = select_child_patients(
                splits, patients_with_kernel(meta["train"], table, kernel) & parent_patients,
                n_train=args.train_patients, n_val=args.val_patients, seed=settings["seed"],
            )
        worklist, summary = build_worklist(
            meta["train"], meta["valid"], kernel_table=table, chunk_size=settings["chunk_size"], train_kernel=kernel,
            max_train_patients=settings["max_train_patients"], max_test_patients=settings["max_test_patients"],
            max_combined_gb=settings["max_combined_gb"], target_spacing_zyx=cfg.preprocess.target_spacing_zyx,
            seed=settings["seed"], train_patients=child_patients,
        )
    print(format_summary(summary, ing.est_mb_per_volume))

    cache_dir = Path(cfg.paths.cache_dir)
    cached = sum(is_cache_fresh(cache_dir, v, cfg.preprocess) for v in worklist["volume_id"])
    to_get = len(worklist) - cached
    free = shutil.disk_usage(cache_dir if cache_dir.exists() else Path(".")).free / 1e9
    print(
        f"plan: {cached} volumes already in the shared cache, {to_get} to download "
        f"(~{to_get * ing.est_raw_gb_per_volume / 1000:.2f} TB download, +~{to_get * ing.est_mb_per_volume / 1000:.0f} GB of cache); "
        f"free disk {free:.0f} GB, ingest stops below {ing.min_free_gb:g} GB"
    )

    run = new_run(cfg.paths, args.source, name)
    write_worklist(worklist, run.worklist_path)
    kernel_table_file = Path(cfg.paths.kernel_table)
    write_run_info(run, {
        "settings": settings, "parent_run": parent.name if parent else None,
        "preprocess_fingerprint": config_fingerprint(cfg.preprocess),
        "kernel_table_sha1": hashlib.sha1(kernel_table_file.read_bytes()).hexdigest()[:12],
        "volumes": len(worklist), "chunks": summary["total_chunks"],
    })
    print(f"wrote {run.worklist_path}")
    print(f"next: python scripts/preprocessing/ingest.py --source {args.source} --run {name} --max-chunks 1   (calibration run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
