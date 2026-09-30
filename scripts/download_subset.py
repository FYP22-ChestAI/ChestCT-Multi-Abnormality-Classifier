"""Step 2: download a CT-RATE subset from Hugging Face -- one combined run.

Not run automatically by anyone else -- this needs YOUR own Hugging Face
login and accepted CT-RATE terms (see README). Run it yourself:

    pip install huggingface_hub
    huggingface-cli login
    python scripts/download_subset.py --n-train 40 --n-val 10 --n-test 10 --seed 0

One command for the whole subset: it fetches n_train+n_val scans from
CT-RATE's official TRAIN pool and n_test scans from CT-RATE's official VALID
pool (CT-RATE's own held-out benchmark set), all in this one run, picked at
random with --seed rather than "the first N alphabetically" so a pilot
subset isn't systematically biased toward one corner of the patient id
range. Pass --n-val 0 and run it again with a different --seed to grow an
existing subset later.

No labels are read or downloaded here, or anywhere in this pipeline (M1) --
only the metadata CSV (acquisition fields: spacing, rows/cols, RescaleSlope,
...), used purely to know which volume ids exist and, optionally, to screen
out ones that would need too much memory to resample. M4 joins labels
separately, by volume_id, when it needs them for supervision.

Fast by construction: this never calls list_repo_files() (which enumerates
the ENTIRE CT-RATE repository -- tens of thousands of files across a deeply
nested per-patient tree -- and was the actual cause of a 10-15 minute wait
before any download even started). Instead, it fetches the small metadata CSV
first (which already lists every real volume name), and builds each file's
exact path directly from CT-RATE's own naming pattern, confirmed against a
second, independently-written CT-RATE loader:

    dataset/{split}_fixed/{split}_{patient}/{split}_{patient}_{scan}/{name}.nii.gz
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

from chestct.preprocessing.config import load_config
from chestct.preprocessing.manifest import _strip_nifti_ext

REPO_ID = "ibrahimhamamci/CT-RATE"
SMALL_FILES = {
    "train": {
        "metadata": "dataset/metadata/train_metadata.csv",
        "no_chest": "dataset/metadata/no_chest_train.txt",
    },
    "valid": {
        "metadata": "dataset/metadata/validation_metadata.csv",
        "no_chest": "dataset/metadata/no_chest_valid.txt",
    },
}
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def _download_flat(hf_hub_download, repo_file: str, dest_dir: Path) -> Path:
    """Download one Hugging Face file into the shared HF cache, then copy it
    to dest_dir under its own filename only -- flattening away the repo's
    "dataset/train_fixed/train_1/train_1_a/..." folder structure."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached_path = hf_hub_download(REPO_ID, filename=repo_file, repo_type="dataset")
    dest_path = dest_dir / Path(repo_file).name
    shutil.copy2(cached_path, dest_path)
    return dest_path


def ctrate_rel_path(volume_id: str, split: str, fixed: bool = True) -> str:
    """The exact per-file relative path -- avoids ever listing the repo.

    The nested "_fixed" layout is confirmed (cross-checked against a second,
    independently-written CT-RATE loader). The plain (non-"_fixed") folder's
    layout was never independently confirmed, so fixed=False falls back to
    a flat guess that may not be correct -- prefer fixed=True.
    """
    m = re.match(rf"^{split}_(\d+)_([a-z]+)_(\d+)$", volume_id)
    if not m:
        raise ValueError(f"unexpected CT-RATE volume id for split {split!r}: {volume_id!r}")
    patient, scan, _ = m.groups()
    if fixed:
        return f"dataset/{split}_fixed/{split}_{patient}/{split}_{patient}_{scan}/{volume_id}.nii.gz"
    return f"dataset/{split}/{volume_id}.nii.gz"


def _num(s) -> float:
    m = _NUM_RE.findall(str(s))
    return float(m[0]) if m else float("nan")


def _passes_screen(volume_id: str, metadata_df, max_combined_gb: float, target_spacing_zyx) -> bool:
    """Estimate the peak memory a resample would need (original + resampled
    array both in memory at once, float32) and reject anything over budget
    -- without downloading it first. This is what fixed the OOM we hit on a
    1024x1024x237 real CT-RATE volume; see docs/data_contract.md."""
    if volume_id not in metadata_df.index:
        return True  # can't screen without metadata -- let it through
    row = metadata_df.loc[volume_id]
    try:
        rows_, cols_, n_slices = int(row.Rows), int(row.Columns), int(row.NumberofSlices)
        xy, z = _num(row.XYSpacing), float(row.ZSpacing)
    except (AttributeError, ValueError, TypeError):
        return True
    tz, ty, tx = target_spacing_zyx
    fy, fx, fz = xy / ty, xy / tx, z / tz
    orig_voxels = rows_ * cols_ * n_slices
    resampled_voxels = (rows_ * fy) * (cols_ * fx) * (n_slices * fz)
    combined_gb = (orig_voxels + resampled_voxels) * 4 / 1e9  # float32, both arrays alive at once
    return combined_gb <= max_combined_gb


def _select_and_download(
    hf_hub_download,
    split: str,
    n_wanted: int,
    seed: int,
    out_dir: Path,
    meta_dir: Path,
    fixed: bool,
    no_chest_filter: bool,
    max_combined_gb: float | None,
    cfg,
) -> None:
    if n_wanted <= 0:
        return
    import pandas as pd

    files = SMALL_FILES[split]
    print(f"[{split}] fetching metadata/no_chest files to {meta_dir} ...")
    metadata_path = _download_flat(hf_hub_download, files["metadata"], meta_dir)
    no_chest_path = _download_flat(hf_hub_download, files["no_chest"], meta_dir)

    metadata_df = pd.read_csv(metadata_path)
    metadata_df["VolumeName"] = metadata_df["VolumeName"].map(_strip_nifti_ext)
    candidate_ids = metadata_df["VolumeName"].tolist()
    print(f"[{split}] {len(candidate_ids)} candidate volumes listed in the metadata CSV")

    if no_chest_filter:
        excluded = {_strip_nifti_ext(line.strip()) for line in no_chest_path.read_text().splitlines() if line.strip()}
        before = len(candidate_ids)
        candidate_ids = [v for v in candidate_ids if v not in excluded]
        print(f"[{split}] dropped {before - len(candidate_ids)} volumes listed in {no_chest_path.name}")

    metadata_indexed = metadata_df.set_index("VolumeName")
    if max_combined_gb:
        before = len(candidate_ids)
        candidate_ids = [
            v for v in candidate_ids if _passes_screen(v, metadata_indexed, max_combined_gb, cfg.preprocess.target_spacing_zyx)
        ]
        print(f"[{split}] screened out {before - len(candidate_ids)} volume(s) estimated over {max_combined_gb} GB peak resample memory")

    if n_wanted > len(candidate_ids):
        raise SystemExit(f"[{split}] asked for {n_wanted} volumes but only {len(candidate_ids)} candidates are available")

    rng = random.Random(seed)
    selected = sorted(rng.sample(candidate_ids, n_wanted))
    print(f"[{split}] downloading {len(selected)} of {len(candidate_ids)} candidate volumes to {out_dir} ...")
    for i, name in enumerate(selected, 1):
        rel = ctrate_rel_path(name, split, fixed=fixed)
        try:
            p = _download_flat(hf_hub_download, rel, out_dir)
            print(f"[{split} {i}/{len(selected)}] {name}  {p.stat().st_size / 1e6:.0f} MB")
        except Exception as exc:  # noqa: BLE001 - keep going on one bad file, report it clearly
            print(f"[{split} {i}/{len(selected)}] {name}  SKIPPED: {exc!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-train", type=int, required=True, help="scans to pull from CT-RATE's official TRAIN pool for training")
    ap.add_argument("--n-val", type=int, required=True, help="scans to pull from CT-RATE's official TRAIN pool, held out for validation")
    ap.add_argument("--n-test", type=int, required=True, help="scans to pull from CT-RATE's official VALID pool (the benchmark set)")
    ap.add_argument("--seed", type=int, default=0, help="random selection seed -- same seed+counts always picks the same volumes")
    ap.add_argument("--out-dir", default="data/raw", help="flat destination for the .nii.gz scans")
    ap.add_argument(
        "--meta-dir", default=None, help="flat destination for metadata/no_chest files (default: data/metadata)"
    )
    ap.add_argument("--no-fixed", action="store_true", help="use the plain (non-'_fixed') CT-RATE folder -- layout unconfirmed, prefer the default")
    ap.add_argument("--no-chest-filter", action="store_true", help="drop volumes listed in the no_chest_*.txt file")
    ap.add_argument(
        "--max-combined-gb",
        type=float,
        default=None,
        help="skip volumes whose estimated resample memory footprint exceeds this (needs the metadata CSV; see docs/data_contract.md)",
    )
    ap.add_argument("--config", default="configs/data.yaml", help="used only to read the target spacing for --max-combined-gb screening")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit("pip install huggingface_hub first, then run: huggingface-cli login") from exc

    out_dir = Path(args.out_dir)
    meta_dir = Path(args.meta_dir) if args.meta_dir else out_dir.parent / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)

    _select_and_download(
        hf_hub_download, "train", args.n_train + args.n_val, args.seed, out_dir, meta_dir,
        fixed=not args.no_fixed, no_chest_filter=args.no_chest_filter, max_combined_gb=args.max_combined_gb, cfg=cfg,
    )
    _select_and_download(
        hf_hub_download, "valid", args.n_test, args.seed, out_dir, meta_dir,
        fixed=not args.no_fixed, no_chest_filter=args.no_chest_filter, max_combined_gb=args.max_combined_gb, cfg=cfg,
    )

    print(f"done. Scans are in {out_dir} (matches configs/data.yaml's default paths.raw_dir). "
          f"Metadata is in {meta_dir}: point scripts/build_manifest.py --metadata at train_metadata.csv, "
          f"with the SAME --n-train/--n-val/--seed used here.")


if __name__ == "__main__":
    main()
