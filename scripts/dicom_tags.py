"""Print what the DICOM tags of a scan (or the first few scans under a root) contain.

    python scripts/dicom_tags.py /content/local_scans/nhrd --max 3

Header-only (no pixel data is decoded), so it is fast. It shows the
acquisition fields, the compression type, whether slice spacing is uniform,
and -- without printing the value -- whether a PatientID is present and a
short hash of it, so you can see whether different folders really carry
different patient ids. Use it on 1-2 real scans first to check what the
anonymiser left in the files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from chestct.preprocessing.dicom_loader import DicomReadError, discover_scans, list_dicom_files, summarize_dicom_folder


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a scan folder, or a root folder containing scans")
    ap.add_argument("--max", type=int, default=3, help="how many scans to describe when given a root")
    args = ap.parse_args()

    root = Path(args.path)
    if list_dicom_files(root):
        targets = [(root.name, root)]
    else:
        targets = [(e.scan_path, root / e.scan_path) for e in discover_scans(root, fmt="dicom", max_scans=args.max)]
    if not targets:
        raise SystemExit(f"no DICOM files found under {root}")

    hashes = []
    for name, folder in targets:
        print(f"=== {name}")
        try:
            summary = summarize_dicom_folder(folder)
        except DicomReadError as exc:
            print(f"  cannot read: {exc}")
            continue
        for key, value in summary.items():
            print(f"  {key}: {value}")
        hashes.append(summary.get("patient_id_hash", ""))

    if len(hashes) > 1:
        distinct = len({h for h in hashes if h})
        print(f"\n{len(hashes)} scans, {distinct} distinct PatientID hash(es) "
              f"({'ids differ between scans' if distinct == len(hashes) else 'ids are missing or shared -- use the folder-depth rule'})")


if __name__ == "__main__":
    main()
