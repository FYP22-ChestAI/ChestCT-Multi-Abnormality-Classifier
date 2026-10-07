"""Step 4: quality-check every cached volume.

    python scripts/preprocessing/qc_report.py --source ctrate --run train-sharp

Re-checks each cached .npy listed in the run's manifest.csv (with the per-source QC
thresholds from the config), writes the run's qc_report.csv (pass/fail, reasons,
HU stats) and montage PNGs for a human to eyeball. Review the failures, then
run assign_splits.py: volumes that failed QC are excluded BEFORE patients are
assigned, so val/test counts mean usable patients.

Separate from ingest on purpose: it can be re-run any time (e.g. after tuning a
threshold in the config) without touching a single raw scan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ct_preprocessing.cli import add_config_arg, friendly_errors, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.pipeline import apply_loader_checks
from ct_preprocessing.quality import check_volume, save_montage
from ct_preprocessing.runs import resolve_run


@friendly_errors
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument(
        "--montage-every", type=int, default=25,
        help="save a montage for 1 in every N volumes, plus every volume that fails a check",
    )
    ap.add_argument("--source", required=True, help="the source of the run to check")
    ap.add_argument("--run", default=None, help="the run to check (default: the only one)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = resolve_run(cfg.paths, args.source, args.run)
    show_settings("using", dict(source=args.source, run=run.name, manifest=run.manifest_path, montage_every=args.montage_every))
    if not run.manifest_path.exists():
        raise FileNotFoundError(f"{run.manifest_path} not found -- run merge_manifests.py --source {args.source} --run {run.name} first")
    manifest = pd.read_csv(run.manifest_path)
    cache_dir = Path(cfg.paths.cache_dir)
    montage_dir = run.montage_dir
    montage_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, (_, row) in enumerate(manifest.iterrows()):
        npy_path = cache_dir / f"{row['volume_id']}.npy"
        if not npy_path.exists():
            rows.append({"volume_id": row["volume_id"], "passed": False, "reasons": "not preprocessed", "flags": ""})
            continue

        arr = np.load(npy_path)
        spacing = (row.get("spacing_z_mm"), row.get("spacing_y_mm"), row.get("spacing_x_mm"))
        source_name = row["source_name"] if "source_name" in row and pd.notna(row["source_name"]) else None
        result = check_volume(row["volume_id"], arr, spacing, cfg.qc_for(source_name))
        sidecar = cache_dir / f"{row['volume_id']}.meta.json"
        if sidecar.exists():  # geometry checks recorded at preprocessing time (DICOM)
            apply_loader_checks(result, json.loads(sidecar.read_text()).get("loader_checks", {}))
        rows.append(
            {
                "volume_id": result.volume_id,
                "passed": result.passed,
                "reasons": "; ".join(result.reasons),
                "flags": "; ".join(result.flags),
                **result.stats,
            }
        )
        if not result.passed or n % args.montage_every == 0:
            save_montage(arr, montage_dir / f"{row['volume_id']}.png")
        if (n + 1) % 1000 == 0:
            print(f"  checked {n + 1}/{len(manifest)}", flush=True)

    qc_df = pd.DataFrame(rows)
    out_path = run.qc_report_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qc_df.to_csv(out_path, index=False)
    print(f"{int(qc_df['passed'].sum())} / {len(qc_df)} passed -> {out_path}")
    print(f"montages saved to {montage_dir}")
    print(f"next: python scripts/preprocessing/assign_splits.py --source {args.source} --run {run.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
