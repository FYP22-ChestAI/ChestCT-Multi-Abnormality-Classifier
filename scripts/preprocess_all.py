"""Steps 5-9: run the preprocessing pipeline over every volume in the manifest.

Resumable: re-running skips a volume only if its cached .npy was produced by
the EXACT SAME config as the one active now (checked via a fingerprint
sidecar, see chestct.preprocessing.preprocess.is_cache_fresh) -- switching CT-RATE
download folders, or changing any preprocessing setting, correctly triggers
a reprocess instead of silently reusing stale, differently-processed output.

Handles both formats: each manifest row says where its scan lives
(`scan_path`, relative to its source's folder) and what it is (`format`).
Where a source's folder is comes from configs/data.yaml, or from
--source-root NAME=PATH given at run time (e.g. a Drive copy in Colab, an SSD
or a server folder) -- the manifest itself never stores an absolute path.

For NIfTI, each volume's own RescaleSlope/RescaleIntercept (from the manifest,
joined in by build_manifest.py from CT-RATE's metadata CSV) is passed through
so the loader can explicitly correct a scan that isn't calibrated yet. DICOM
carries its own values in every slice and needs nothing extra.

Usage:
    python scripts/preprocess_all.py --config configs/data.yaml --workers 4
    python scripts/preprocess_all.py --limit 50   # pilot run, Step 2-5 sanity check
    python scripts/preprocess_all.py --device auto --workers 1  # use a GPU if present
    python scripts/preprocess_all.py --source-root nhrd_local=/content/local_scans/nhrd
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from chestct.preprocessing.config import load_config
from chestct.preprocessing.preprocess import PreprocessConfig, config_fingerprint, is_cache_fresh, preprocess_one

try:
    from tqdm import tqdm
except ImportError:  # optional dependency
    def tqdm(iterable, **kwargs):
        return iterable


def _run_one(job: tuple):
    scan_path, out_dir, cfg, volume_id, rescale_slope, rescale_intercept, scan_format, qc_thresholds, series_uid = job
    return preprocess_one(
        scan_path, out_dir, cfg, volume_id=volume_id, rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept, scan_format=scan_format, qc_thresholds=qc_thresholds,
        series_uid=series_uid,
    )


def _parse_source_roots(items: list[str]) -> dict[str, Path]:
    roots = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--source-root expects NAME=PATH, got {item!r}")
        roots[name.strip()] = Path(path.strip())
    return roots


def _first_present(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="reprocess even if a fresh cache already exists")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N manifest rows (pilot run)")
    ap.add_argument("--device", default=None, help="overrides configs/data.yaml's preprocess.device (cpu/auto/cuda)")
    ap.add_argument(
        "--source-root", action="append", default=[], metavar="NAME=PATH",
        help="where a source's scans live for THIS run (repeatable); overrides the source's raw_dir in configs/data.yaml",
    )
    args = ap.parse_args()
    source_roots = _parse_source_roots(args.source_root)

    cfg = load_config(args.config)
    preprocess_cfg = cfg.preprocess if args.device is None else replace(cfg.preprocess, device=args.device)

    manifest = pd.read_csv(cfg.paths.manifest_path)
    if args.limit:
        manifest = manifest.head(args.limit)

    raw_dir = Path(cfg.paths.raw_dir)
    cache_dir = Path(cfg.paths.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for _, row in manifest.iterrows():
        source_name = row["source_name"] if "source_name" in row and pd.notna(row["source_name"]) else None
        if source_name in source_roots:
            this_raw_dir = source_roots[source_name]
        elif source_name in cfg.sources:
            this_raw_dir = Path(cfg.sources[source_name].raw_dir)
        else:
            this_raw_dir = raw_dir
        rel = row["scan_path"] if "scan_path" in row and pd.notna(row["scan_path"]) else f"{row['volume_id']}.nii.gz"
        scan_format = row["format"] if "format" in row and pd.notna(row["format"]) else "nifti"
        scan_path = this_raw_dir / rel
        series_uid = row["series_uid"] if "series_uid" in row and pd.notna(row["series_uid"]) else None

        if not args.force and scan_path.exists() and is_cache_fresh(
            cache_dir, row["volume_id"], preprocess_cfg, scan_path=scan_path, scan_format=scan_format,
            series_uid=series_uid,
        ):
            continue
        rescale_slope = _first_present(row, ["RescaleSlope"])
        rescale_intercept = _first_present(row, ["RescaleIntercept"])
        jobs.append((
            str(scan_path), str(cache_dir), preprocess_cfg, row["volume_id"], rescale_slope, rescale_intercept,
            scan_format, cfg.qc_for(source_name), series_uid,
        ))

    print(f"{len(jobs)} volumes to process ({len(manifest) - len(jobs)} already cached and up to date)")
    if source_roots:
        print("source roots for this run:", {k: str(v) for k, v in source_roots.items()})
    print(f"device: {preprocess_cfg.device}, config fingerprint (see stale-cache guard): "
          f"{config_fingerprint(preprocess_cfg)}")

    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_run_one, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            results.append(fut.result())

    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    print(f"done: {len(ok)} ok, {len(failed)} failed, {time.time() - t0:.1f}s total")
    if ok:
        avg_s = sum(r.seconds for r in ok) / len(ok)
        avg_mb = sum(Path(r.out_path).stat().st_size for r in ok) / len(ok) / 1e6
        n_qc_failed = sum(1 for r in ok if r.qc_passed is False)
        print(f"avg {avg_s:.2f}s/volume, avg {avg_mb:.1f} MB/volume on disk")
        if n_qc_failed:
            print(f"{n_qc_failed} volume(s) saved but flagged by QC -- see scripts/qc_report.py for details")
    for r in failed:
        print(f"  FAILED {r.volume_id}: {r.error}")

    # Fold per-volume stats (size, spacing after resample, crop shape) back
    # into manifest.csv, so downstream steps and analysis don't need to
    # re-open every .npy just to know its shape or spacing.
    if ok:
        stats_rows = []
        for r in ok:
            sz, sy, sx = r.spacing_after_resample
            stats_rows.append(
                {
                    "volume_id": r.volume_id,
                    "n_slices": r.n_slices,
                    "npy_path": r.out_path,
                    "spacing_z_mm": sz,
                    "spacing_y_mm": sy,
                    "spacing_x_mm": sx,
                    "crop_shape": "x".join(map(str, r.crop_shape)),
                    "qc_passed": r.qc_passed,
                }
            )
        stats_df = pd.DataFrame(stats_rows)

        full_manifest = pd.read_csv(cfg.paths.manifest_path)
        overlap_cols = [c for c in stats_df.columns if c != "volume_id" and c in full_manifest.columns]
        full_manifest = full_manifest.drop(columns=overlap_cols)
        full_manifest = full_manifest.merge(stats_df, on="volume_id", how="left")
        full_manifest.to_csv(cfg.paths.manifest_path, index=False)
        print(f"updated {cfg.paths.manifest_path} with per-volume preprocessing stats")

    manifest_out = {
        "config": asdict(preprocess_cfg),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "failed": [{"volume_id": r.volume_id, "error": r.error} for r in failed],
    }
    out_path = Path(cfg.paths.preprocessing_manifest_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest_out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
