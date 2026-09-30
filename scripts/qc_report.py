"""Step 8: run quality checks over every cached .npy, save qc_report.csv and
montage PNGs for eyeballing.

Usage:
    python scripts/qc_report.py --config configs/data.yaml --montage-every 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import json

import numpy as np
import pandas as pd

from chestct.preprocessing.config import load_config
from chestct.preprocessing.pipeline import apply_loader_checks
from chestct.preprocessing.quality import check_volume, save_montage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument(
        "--montage-every",
        type=int,
        default=25,
        help="save a montage for 1 in every N scans, plus every scan that fails a check",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest = pd.read_csv(cfg.paths.manifest_path)
    cache_dir = Path(cfg.paths.cache_dir)
    montage_dir = Path(cfg.paths.montage_dir)
    montage_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in manifest.iterrows():
        npy_path = cache_dir / f"{row['volume_id']}.npy"
        if not npy_path.exists():
            rows.append({"volume_id": row["volume_id"], "passed": False, "reasons": "not preprocessed yet", "flags": ""})
            continue

        arr = np.load(npy_path)
        spacing = (row.get("spacing_z_mm"), row.get("spacing_y_mm"), row.get("spacing_x_mm"))
        source_name = row["source_name"] if "source_name" in row and pd.notna(row["source_name"]) else None
        result = check_volume(row["volume_id"], arr, spacing, cfg.qc_for(source_name))
        sidecar = cache_dir / f"{row['volume_id']}.meta.json"
        if sidecar.exists():  # DICOM geometry checks recorded at preprocessing time
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

        if not result.passed or i % args.montage_every == 0:
            save_montage(arr, montage_dir / f"{row['volume_id']}.png")

    qc_df = pd.DataFrame(rows)
    out_path = Path(cfg.paths.qc_report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qc_df.to_csv(out_path, index=False)
    print(f"{int(qc_df['passed'].sum())} / {len(qc_df)} passed -> {out_path}")
    print(f"montages saved to {montage_dir}")


if __name__ == "__main__":
    main()
