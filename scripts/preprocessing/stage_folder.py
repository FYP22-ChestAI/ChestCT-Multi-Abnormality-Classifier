"""Copy scans from a slow location (e.g. a mounted Google Drive) to local disk.

    python scripts/stage_folder.py --src "/content/drive/MyDrive/NHRD - CT Scan" \\
        --dst /content/local_scans/nhrd --only-folders 4203-26,4214-26

Resumable: files already copied (same size) are skipped, so it is safe to
re-run. The rest of the pipeline then reads from --dst.
"""
from __future__ import annotations

import argparse
import time

from ct_preprocessing.staging import stage_tree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the source folder (any path, e.g. a mounted Drive)")
    ap.add_argument("--dst", required=True, help="local destination folder")
    ap.add_argument("--only-folders", default="", help="comma-separated top-level folders to copy (default: all)")
    ap.add_argument("--workers", type=int, default=8, help="parallel copies (Drive latency is what makes this slow)")
    args = ap.parse_args()

    only = [f for f in args.only_folders.split(",") if f.strip()]
    t0 = time.time()
    counts = stage_tree(args.src, args.dst, only_folders=only or None, workers=args.workers)
    print(f"{counts['files']} files: {counts['copied']} copied, {counts['skipped']} already there "
          f"({counts['bytes'] / 1e6:.0f} MB, {time.time() - t0:.0f}s) -> {args.dst}")


if __name__ == "__main__":
    main()
