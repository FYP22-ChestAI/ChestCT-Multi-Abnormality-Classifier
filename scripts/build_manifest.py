"""Step 3: build/extend data/manifest.csv from one named data source.

No labels are ever read or joined here -- for CT-RATE or for local data. This
pipeline (M1) only ever produces images, patient ids and splits; M4 joins
labels on its own side, by volume_id, when it needs them for supervision. See
docs/data_contract.md.

A "source" is one entry under `sources:` in configs/data.yaml. Run this once
per source, with --append after the first, to accumulate all of them into one
unified manifest.csv, distinguished by the `source_name` column.

With no --n-train/--n-val/--n-test/--seed at all, each source's own
configs/data.yaml defaults are used (a small pilot size out of the box).
Override any of them to change the amount or reproduce a different split.

CT-RATE (NIfTI, ids like train_1_a_1) -- point --raw-dir at whatever
scripts/download_subset.py just downloaded, and pass the SAME n-train/n-val
and seed so the val cut is reproducible:

    python scripts/build_manifest.py --source-name ctrate \\
        --builder ctrate --raw-dir data/raw --metadata data/metadata/train_metadata.csv

Any folder of DICOM series or NIfTI files -- e.g. the local NHRD data, on a
Drive mount, an SSD or a server; the folder is given at run time and can have
any layout (every DICOM series -- grouped by SeriesInstanceUID, not just
folder location -- or NIfTI file is one scan). Patients are grouped by the
DICOM PatientID tag or by folder depth -- "auto" (the default) decides for
itself, from whether the tag actually survived any anonymisation:

    python scripts/build_manifest.py --source-name nhrd_local --append \\
        --builder folder --raw-dir /content/local_scans/nhrd

Splitting is always by PATIENT, at a fixed amount (not a ratio) so the same
--n-train/--n-val/--n-test/--seed on the same raw-dir always reproduces the
same split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from chestct.preprocessing.config import SourceConfig, load_config
from chestct.preprocessing.dicom_loader import discover_scans
from chestct.preprocessing.manifest import (
    FOLDER_BUILDER,
    assign_splits_by_amount,
    build_manifest,
    build_manifest_folder,
    check_no_patient_overlap,
)

_BAD_SOURCE_NAME_CHARS = set(" \t\n\r\"'")


def validate_source_name(name: str) -> str:
    """A source_name is used both as a configs/data.yaml key and as a bare
    token dropped into shell commands by the notebooks -- a space in it once
    silently split into two command-line arguments there. A leading/trailing
    space is almost always just a stray keystroke in a form field, so it's
    trimmed rather than rejected; a space (or quote) still inside the name
    after trimming is much more likely a real typo, so that still stops here
    rather than silently becoming a different (wrong) config-lookup key or
    manifest.csv value downstream."""
    name = name.strip()
    if not name:
        raise SystemExit("--source-name must not be empty")
    if _BAD_SOURCE_NAME_CHARS & set(name):
        raise SystemExit(f"--source-name {name!r} must not contain spaces or quote characters")
    return name


def find_volume_ids(raw_dir: Path) -> list[str]:
    return sorted(p.name.split(".")[0] for p in raw_dir.glob("*.nii.gz"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-name", default="ctrate", help="a name under configs/data.yaml's `sources:`")
    ap.add_argument("--raw-dir", default=None, help="the folder holding the scans; overrides the source's configured raw_dir")
    ap.add_argument("--format", default=None, help="nifti, dicom or auto; overrides the source's configured format")
    ap.add_argument("--builder", default=None, help="'ctrate' or 'folder'; overrides the source's manifest_builder")
    ap.add_argument("--metadata", default=None, help="CT-RATE metadata csv (optional; acquisition fields only, never labels)")
    ap.add_argument("--patient-path-depth", type=int, default=None, help="folder sources: folder levels that identify a patient (used only as a fallback when the DICOM PatientID tag is absent)")
    ap.add_argument("--patient-id-source", default=None, help="folder sources: 'auto' (default), 'path' or 'dicom_tag'")
    ap.add_argument("--n-train", type=int, default=None, help="patients assigned to train (default: this source's configs/data.yaml n_train)")
    ap.add_argument("--n-val", type=int, default=None, help="patients assigned to val (default: this source's configs/data.yaml n_val)")
    ap.add_argument("--n-test", type=int, default=None, help="folder sources: patients assigned to test (CT-RATE: its own valid-source patients become test directly, so leave this 0; default: this source's configs/data.yaml n_test, else 0)")
    ap.add_argument("--seed", type=int, default=None, help="default: this source's configs/data.yaml seed, else 0")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--append", action="store_true", help="append to an existing manifest.csv instead of overwriting")
    args = ap.parse_args()

    args.source_name = validate_source_name(args.source_name)

    cfg = load_config(args.config)
    source_cfg = cfg.sources.get(args.source_name)
    if source_cfg is None:
        print(f"note: {args.source_name!r} is not in configs/data.yaml's `sources:` -- using defaults and command-line values")
        source_cfg = SourceConfig(raw_dir=cfg.paths.raw_dir, format=args.format or "auto")

    raw_dir = Path(args.raw_dir or source_cfg.raw_dir)
    fmt = args.format or source_cfg.format
    builder = args.builder or (source_cfg.manifest_builder if args.source_name in cfg.sources else FOLDER_BUILDER)

    n_train = args.n_train if args.n_train is not None else source_cfg.n_train
    n_val = args.n_val if args.n_val is not None else source_cfg.n_val
    n_test = args.n_test if args.n_test is not None else (source_cfg.n_test if source_cfg.n_test is not None else 0)
    seed = args.seed if args.seed is not None else source_cfg.seed
    if n_train is None or n_val is None:
        raise SystemExit(
            f"no --n-train/--n-val given, and no default is set in configs/data.yaml's "
            f"sources.{args.source_name}.n_train/n_val -- pass them explicitly, or add defaults to the config"
        )
    print(f"using: n_train={n_train}, n_val={n_val}, n_test={n_test}, seed={seed} (override any of these with --n-train etc.)")

    if builder == FOLDER_BUILDER:
        if args.metadata:
            print("note: --metadata is not used for folder sources")
        entries = discover_scans(raw_dir, fmt=fmt)
        if not entries:
            raise SystemExit(f"no scans found under {raw_dir} (format={fmt})")
        print(f"found {len(entries)} scan(s) under {raw_dir}")
        manifest = build_manifest_folder(
            raw_dir,
            entries,
            patient_depth=args.patient_path_depth or source_cfg.patient_path_depth,
            patient_id_source=args.patient_id_source or source_cfg.patient_id_source,
        )
        patients = manifest["patient_id"].unique().tolist()
        split_map = assign_splits_by_amount(patients, n_train, n_val, n_test, seed=seed)
        manifest["split"] = manifest["patient_id"].map(split_map)
        before = len(manifest)
        manifest = manifest[manifest["split"].notna()].copy()
        if before - len(manifest):
            print(f"dropped {before - len(manifest)} scan(s) whose patient was not selected by --n-train/--n-val/--n-test")
    else:
        if fmt not in ("nifti", "auto"):
            raise SystemExit(f"the {builder!r} builder only handles NIfTI (format={fmt!r})")
        volume_ids = find_volume_ids(raw_dir)
        if not volume_ids:
            raise SystemExit(f"no .nii.gz files found under {raw_dir}")

        metadata_df = pd.read_csv(args.metadata) if args.metadata else None
        manifest = build_manifest(volume_ids, metadata_df, builder=builder)

        # CT-RATE's own valid-split patients are always the held-out benchmark
        # set (proposal 4.2.1) -- never mixed into train/val.
        train_mask = manifest["source_split"] == "train"
        train_patients = manifest.loc[train_mask, "patient_id"].unique().tolist()
        split_map = assign_splits_by_amount(train_patients, n_train, n_val, n_test=0, seed=seed)
        manifest["split"] = manifest["patient_id"].map(split_map)
        manifest.loc[manifest["source_split"] == "valid", "split"] = "test"
        before = len(manifest)
        manifest = manifest[manifest["split"].notna()].copy()
        if before - len(manifest):
            print(f"dropped {before - len(manifest)} train-source scan(s) whose patient was not selected by --n-train/--n-val")

    manifest["source_name"] = args.source_name

    out_path = Path(cfg.paths.manifest_path)
    if args.append and out_path.exists():
        existing = pd.read_csv(out_path)
        manifest = pd.concat([existing, manifest], ignore_index=True)
        manifest = manifest.drop_duplicates(subset="volume_id", keep="last")

    check_no_patient_overlap(manifest)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    print(f"wrote {len(manifest)} rows to {out_path} (source_name={args.source_name!r})")
    print(manifest.groupby("source_name")["split"].value_counts().to_string())
    if "manifest_problem" in manifest.columns:
        problems = manifest[manifest["manifest_problem"].notna()]
        for _, row in problems.iterrows():
            print(f"  PROBLEM {row['volume_id']}: {row['manifest_problem']}")


if __name__ == "__main__":
    main()
