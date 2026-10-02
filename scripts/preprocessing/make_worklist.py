"""Step 1 (CT-RATE only): decide exactly which volumes to ingest.

    python scripts/preprocessing/make_worklist.py

Downloads CT-RATE's two small metadata CSVs (~16 MB, no images), then writes
data/worklists/ctrate.csv: every volume to fetch, grouped into chunks. This is
a work list, NOT a split -- train/val/test is decided later, after ingest and
QC (assign_splits.py). The whole valid_fixed pool is taken (the test set); the
train pool is taken one reconstruction per scan, in a seeded random order of
patients, optionally capped. Chunks of the test pool come first, then train.

It refuses to overwrite an existing worklist unless --force, and even then
only if no already-ingested chunk would change. Every default comes from
configs/preprocessing.yaml (sources.ctrate.ingest); override any of them here.
Needs your own Hugging Face login: `huggingface-cli login`, after accepting
the CT-RATE terms on the dataset page.
"""
from __future__ import annotations

import argparse

from ct_preprocessing.cli import add_config_arg, friendly_errors, pick, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.ingest import state
from ct_preprocessing.ingest.ctrate import download_metadata, load_metadata
from ct_preprocessing.ingest.sources import get_hf_download
from ct_preprocessing.ingest.worklist import WorklistConflict, build_worklist, format_summary, write_worklist
from ct_preprocessing.manifest import CTRATE_BUILDER


@friendly_errors
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--source", default="ctrate", help="a CT-RATE-type source in the config (default: ctrate)")
    ap.add_argument("--chunk-size", type=int, default=None, help="volumes per chunk")
    ap.add_argument("--train-pool", choices=["one_per_scan", "all"], default=None, help="which reconstructions of the train pool to keep")
    ap.add_argument("--test-pool", choices=["one_per_scan", "all"], default=None, help="which reconstructions of the valid pool to keep")
    ap.add_argument("--max-train-patients", type=int, default=None, help="cap the train pool to this many patients")
    ap.add_argument("--max-test-patients", type=int, default=None, help="cap the test pool (for small pilot runs only)")
    ap.add_argument("--max-combined-gb", type=float, default=None, help="skip TRAIN volumes needing more resample memory than this")
    ap.add_argument("--seed", type=int, default=None, help="patient shuffle seed")
    ap.add_argument("--refresh-metadata", action="store_true", help="download the metadata CSVs again")
    ap.add_argument("--force", action="store_true", help="rebuild an existing worklist (only if no ingested chunk changes)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source_cfg = cfg.source(args.source)
    if source_cfg.manifest_builder != CTRATE_BUILDER:
        raise SystemExit(f"{args.source!r} is not a CT-RATE-type source -- only those have a worklist")
    ing = source_cfg.ingest
    settings = dict(
        source=args.source,
        chunk_size=pick(args.chunk_size, ing.chunk_size),
        train_pool=pick(args.train_pool, ing.train_pool),
        test_pool=pick(args.test_pool, ing.test_pool),
        max_train_patients=pick(args.max_train_patients, ing.max_train_patients),
        max_test_patients=pick(args.max_test_patients, ing.max_test_patients),
        max_combined_gb=pick(args.max_combined_gb, ing.max_combined_gb),
        seed=pick(args.seed, ing.seed),
    )
    show_settings("using", settings)

    out = state.worklist_path(cfg.paths, args.source)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists -- it is reused as is. Pass --force to rebuild it.")

    download_metadata(cfg.paths, ing.hf_repo, get_hf_download(), refresh=args.refresh_metadata)
    meta = load_metadata(cfg.paths)
    worklist, summary = build_worklist(
        meta["train"], meta["valid"],
        chunk_size=settings["chunk_size"], train_pool=settings["train_pool"], test_pool=settings["test_pool"],
        max_train_patients=settings["max_train_patients"], max_test_patients=settings["max_test_patients"],
        max_combined_gb=settings["max_combined_gb"],
        target_spacing_zyx=cfg.preprocess.target_spacing_zyx, seed=settings["seed"],
    )
    try:
        write_worklist(worklist, out, overwrite=args.force, done_chunks=set(state.done_chunks(cfg.paths, args.source)))
    except WorklistConflict as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(format_summary(summary, ing.est_mb_per_volume))
    print(f"wrote {out}")
    print("next: python scripts/preprocessing/ingest.py --source " + args.source + " --max-chunks 1   (calibration run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
