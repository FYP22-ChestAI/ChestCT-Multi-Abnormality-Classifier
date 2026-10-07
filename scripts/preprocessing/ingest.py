"""Step 2: ingest a run chunk by chunk -- the continuous script.

    python scripts/preprocessing/ingest.py --source ctrate --run train-sharp
    python scripts/preprocessing/ingest.py --source nhrd_local

For every pending chunk: fetch its raw scans, preprocess them into the shared cache
(data/cache/), record its manifest rows in the run's folder, delete the raw data, write a .done
marker. Volumes any earlier run already cached are NOT fetched again. Raw data never accumulates,
so a dataset far bigger than the disk can be processed. Run it inside tmux and walk away:

    tmux new -s ingest        # detach with Ctrl+b d, reattach with: tmux a -t ingest

Safe to stop and re-run at any point -- finished chunks are skipped, and a half-finished one resumes
without re-fetching what is already cached. It stops cleanly (never crashes the server) when free disk
falls below --min-free-gb, and stops after several chunks in a row fail (a network or login problem).

  * ctrate    -- chunks come from the run's worklist.csv (make_worklist.py); volumes are downloaded
                 from Hugging Face.
  * nhrd_local -- every .zip/.tar archive found in the configured Drive folder
                 (sources.nhrd_local.ingest.drive_remote) is one chunk. Its run is called "main".

--run NAME picks the run; with only one run on disk it is chosen for you.

First run it with --max-chunks 1 as a calibration: it measures the real cache size per volume and
download speed before you commit to the long run.

The cache remembers which preprocessing settings built it; if the settings in the config changed, ingest
refuses to write into it (it would overwrite volumes earlier runs point to) unless you pass
--allow-new-settings.

Exit code: 0 finished or stopped by a limit you set; 1 some chunk failed; 2 stopped early (low disk, or
repeated failures).
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from ct_preprocessing.cli import add_config_arg, friendly_errors, pick, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.ingest.engine import IngestOptions, ingest_status, run_ingest
from ct_preprocessing.ingest.sources import build_fetcher
from ct_preprocessing.runs import DEFAULT_ARCHIVE_RUN, resolve_run
from ct_preprocessing.manifest import CTRATE_BUILDER


@friendly_errors
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--source", required=True, help="a source name from the config (ctrate, nhrd_local, ...)")
    ap.add_argument("--run", default=None, help="the run to ingest (default: the only one)")
    ap.add_argument("--max-chunks", type=int, default=None, help="stop after this many chunks (e.g. 1 for a calibration run)")
    ap.add_argument("--chunk", default=None, help="process only this one chunk id")
    ap.add_argument("--min-free-gb", type=float, default=None, help="stop when free disk drops below this (default: config)")
    ap.add_argument("--workers", type=int, default=None, help="parallel preprocessing processes (default: config)")
    ap.add_argument("--device", default=None, help="cpu, cuda or auto; overrides preprocess.device (use --workers 1 on a GPU)")
    ap.add_argument("--drive-remote", default=None, help="archive sources: an rclone remote or a folder path (default: config)")
    ap.add_argument("--max-consecutive-failures", type=int, default=3, help="stop after this many failed chunks in a row")
    ap.add_argument("--retry-failed", action="store_true", help="re-run chunks that finished with failed volumes")
    ap.add_argument("--allow-new-settings", action="store_true", help="let changed preprocessing settings replace the cache's recorded ones")
    ap.add_argument("--dry-run", action="store_true", help="list the pending chunks and exit")
    ap.add_argument("--status", action="store_true", help="print progress and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    source_cfg = cfg.source(args.source)
    ing = source_cfg.ingest
    if args.device:
        cfg.preprocess = replace(cfg.preprocess, device=args.device)
    archive_source = source_cfg.manifest_builder != CTRATE_BUILDER
    run = resolve_run(cfg.paths, args.source, args.run, create_default=DEFAULT_ARCHIVE_RUN if archive_source else None)
    options = IngestOptions(
        max_chunks=args.max_chunks,
        only_chunk=args.chunk,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
        min_free_gb=pick(args.min_free_gb, ing.min_free_gb),
        workers=pick(args.workers, ing.workers),
        max_consecutive_failures=args.max_consecutive_failures,
        allow_new_settings=args.allow_new_settings,
    )
    show_settings("using", dict(
        source=args.source, run=run.name, min_free_gb=options.min_free_gb, workers=options.workers,
        device=cfg.preprocess.device, max_chunks=options.max_chunks, scratch=source_cfg.raw_dir,
        cache=cfg.paths.cache_dir,
    ))

    fetcher = build_fetcher(run, cfg, remote=args.drive_remote)
    if args.status:
        print(ingest_status(run, cfg, fetcher))
        return 0

    summary = run_ingest(run, cfg, fetcher, options)
    if summary.stopped and "max-chunks" not in summary.stopped:
        return 2
    return 1 if summary.chunks_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
