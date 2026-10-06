"""Draft the sharp / soft kernel table from CT-RATE's metadata.

    python scripts/preprocessing/make_kernel_table.py
    python scripts/preprocessing/make_kernel_table.py --metadata path/to/train_metadata.csv --out configs/kernel_classes.csv

For every (manufacturer, kernel) pair in the metadata, guesses whether it is a sharp (lung /
high-resolution) or a soft (mediastinal / standard) reconstruction, from the series description
(HRCT, parenchyma, lung, mediastinum) and the kernel name (Siemens Bl = lung, Br number...).
The guess is a STARTING POINT: the file it writes is what worklists really use, so have it checked
-- by the kernel survey (make_worklist.py --survey, then kernel_survey.py), which measures which
reconstruction of each scan is sharper, and by a radiologist or physicist. Change a class by editing
the CSV; a pair that is not in the file counts as "other" and is never chosen as sharp or soft.

Add pairs from other scanners (e.g. NHRD's Toshiba FC81) with --add MANUFACTURER:KERNEL:CLASS.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ct_preprocessing.cli import add_config_arg, friendly_errors, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.kernels import CLASSES, draft_class, normalize_kernel


@friendly_errors
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--metadata", default=None, help="CT-RATE train metadata CSV (default: <metadata_dir>/train_metadata.csv)")
    ap.add_argument("--out", default=None, help="where to write the table (default: paths.kernel_table)")
    ap.add_argument("--add", action="append", default=[], metavar="MANUFACTURER:KERNEL:CLASS", help="an extra pair, e.g. TOSHIBA:FC81:sharp")
    ap.add_argument("--force", action="store_true", help="overwrite an existing table (it may hold reviewed changes)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    metadata = Path(args.metadata or Path(cfg.paths.metadata_dir) / "train_metadata.csv")
    out = Path(args.out or cfg.paths.kernel_table)
    show_settings("using", dict(metadata=metadata, out=out))
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists and may hold reviewed changes -- pass --force to overwrite it")
    if not metadata.exists():
        raise SystemExit(f"{metadata} not found -- run make_worklist.py first (it downloads the metadata) or pass --metadata")

    meta = pd.read_csv(metadata)
    meta["kernel"] = meta["ConvolutionKernel"].map(normalize_kernel)
    meta["draft"] = [
        draft_class(m, k, d) for m, k, d in zip(meta["Manufacturer"], meta["kernel"], meta["SeriesDescription"].fillna(""))
    ]
    rows = []
    for (manufacturer, kernel), group in meta.groupby(["Manufacturer", "kernel"]):
        votes = group["draft"].value_counts()
        rows.append({
            "manufacturer": manufacturer,
            "kernel": kernel,
            "class": votes.index[0],
            "n_volumes": len(group),
            "agreement": round(votes.iloc[0] / len(group), 3),
            "common_description": group["SeriesDescription"].fillna("(empty)").value_counts().index[0],
            "status": "draft",
        })
    for spec in args.add:
        try:
            manufacturer, kernel, klass = spec.split(":")
        except ValueError:
            raise SystemExit(f"--add {spec!r}: expected MANUFACTURER:KERNEL:CLASS")
        if klass not in CLASSES:
            raise SystemExit(f"--add {spec!r}: class must be one of {list(CLASSES)}")
        rows.append({"manufacturer": manufacturer, "kernel": kernel, "class": klass, "n_volumes": "", "agreement": "",
                     "common_description": "", "status": "unverified (added by hand)"})
    table = pd.DataFrame(rows).sort_values("n_volumes", ascending=False, key=lambda s: pd.to_numeric(s, errors="coerce").fillna(-1))
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)

    counts = table.groupby("class")["n_volumes"].apply(lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum()))
    print(f"wrote {len(table)} pairs to {out}: " + ", ".join(f"{k}={v} volumes" for k, v in counts.items()))
    print("next: have it checked (kernel survey / a radiologist), then python scripts/preprocessing/make_worklist.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
