"""CT-RATE as an ingest source: exact-path downloads from Hugging Face.

Nothing here ever lists the repository (tens of thousands of files across a
deeply nested per-patient tree -- that alone once cost a 10-15 minute wait).
Each volume's path follows from its id:

    dataset/{pool}_fixed/{pool}_{patient}/{pool}_{patient}_{scan}/{volume_id}.nii.gz

and the two small metadata CSVs (one per official pool) list every volume
that exists. CT-RATE's train_fixed/ pool feeds train+val; its valid_fixed/
pool is the held-out test set.
"""
from __future__ import annotations

import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import PathsConfig, SourceConfig
from ..manifest import build_manifest_ctrate, strip_nifti_ext
from .base import FetchError, FetchReport

METADATA_FILES = {
    "train": "dataset/metadata/train_metadata.csv",
    "valid": "dataset/metadata/validation_metadata.csv",
}
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
_sleep = time.sleep  # patched in tests so retries do not really wait


def ctrate_rel_path(volume_id: str, pool: str, fixed: bool = True) -> str:
    """The exact per-file repo path -- avoids ever listing the repo.

    The nested "_fixed" layout is confirmed against the real repo. The plain
    (non-"_fixed") folder's layout was never independently confirmed, so
    fixed=False falls back to a flat guess -- prefer fixed=True.
    """
    m = re.match(rf"^{pool}_(\d+)_([a-z]+)_(\d+)$", volume_id)
    if not m:
        raise ValueError(f"unexpected CT-RATE volume id for pool {pool!r}: {volume_id!r}")
    patient, scan, _ = m.groups()
    if fixed:
        return f"dataset/{pool}_fixed/{pool}_{patient}/{pool}_{patient}_{scan}/{volume_id}.nii.gz"
    return f"dataset/{pool}/{volume_id}.nii.gz"


def parse_number(s) -> float:
    """First number in a value like '[0.91796875, 0.91796875]' (XYSpacing is stored as a list string)."""
    m = _NUM_RE.findall(str(s))
    return float(m[0]) if m else float("nan")


def passes_memory_screen(volume_id: str, metadata_indexed: pd.DataFrame, max_combined_gb: float, target_spacing_zyx) -> bool:
    """Estimate the peak memory a resample would need (original + resampled
    array alive at once, float32) and reject anything over budget -- from the
    metadata alone, without downloading. This is what avoids the OOM once hit
    on a 1024x1024x237 volume. Volumes with unusable metadata are let through."""
    if volume_id not in metadata_indexed.index:
        return True
    row = metadata_indexed.loc[volume_id]
    try:
        rows_, cols_, n_slices = int(row.Rows), int(row.Columns), int(row.NumberofSlices)
        xy, z = parse_number(row.XYSpacing), float(row.ZSpacing)
    except (AttributeError, ValueError, TypeError):
        return True
    tz, ty, tx = target_spacing_zyx
    fy, fx, fz = xy / ty, xy / tx, z / tz
    orig_voxels = rows_ * cols_ * n_slices
    resampled_voxels = (rows_ * fy) * (cols_ * fx) * (n_slices * fz)
    return (orig_voxels + resampled_voxels) * 4 / 1e9 <= max_combined_gb


# ------------------------------------------------------------------ metadata CSVs
def metadata_paths(paths: PathsConfig) -> dict[str, Path]:
    return {pool: Path(paths.metadata_dir) / Path(repo_file).name for pool, repo_file in METADATA_FILES.items()}


def download_metadata(paths: PathsConfig, repo_id: str, hf_hub_download: Callable, *, refresh: bool = False) -> dict[str, Path]:
    """Fetch both metadata CSVs (~16 MB, no images) unless already present."""
    out = metadata_paths(paths)
    for pool, dest in out.items():
        if dest.exists() and not refresh:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo_id=repo_id, filename=METADATA_FILES[pool], repo_type="dataset")
        shutil.copyfile(cached, dest)
    return out


def load_metadata(paths: PathsConfig) -> dict[str, pd.DataFrame]:
    out = {}
    for pool, p in metadata_paths(paths).items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found -- run scripts/preprocessing/make_worklist.py first (it downloads it)")
        df = pd.read_csv(p)
        df["VolumeName"] = df["VolumeName"].map(strip_nifti_ext)
        out[pool] = df
    return out


# ---------------------------------------------------------------------- the fetcher
def download_volume(hf_hub_download: Callable, repo_id: str, repo_path: str, scratch_cache: Path, dest: Path) -> None:
    """Download one volume straight into ``dest``.

    huggingface_hub keeps every download in a cache AND would leave it there
    after we delete our copy, so 20,000 volumes would fill the disk. Pointing
    its cache at a scratch folder on the same filesystem and *moving* (a
    rename, no second copy) the file out of it avoids both problems.
    """
    cached = hf_hub_download(repo_id=repo_id, filename=repo_path, repo_type="dataset", cache_dir=str(scratch_cache))
    shutil.move(os.path.realpath(cached), str(dest))


class CtrateFetcher:
    def __init__(
        self,
        worklist: pd.DataFrame,
        metadata: dict[str, pd.DataFrame],
        source_cfg: SourceConfig,
        hf_hub_download: Callable,
        *,
        retries: int = 2,
    ):
        self.worklist = worklist
        self.metadata = pd.concat(list(metadata.values()), ignore_index=True)
        self.cfg = source_cfg
        self.hf_hub_download = hf_hub_download
        self.retries = retries

    @staticmethod
    def _label(chunk) -> str:
        return f"{int(chunk):04d}"

    def _chunk_rows(self, chunk_id: str) -> pd.DataFrame:
        rows = self.worklist[self.worklist["chunk"].map(self._label) == chunk_id]
        if rows.empty:
            raise FetchError(f"chunk {chunk_id} is not in the worklist")
        return rows

    def chunk_ids(self) -> list[str]:
        return [self._label(c) for c in sorted(self.worklist["chunk"].unique())]

    def _download_with_retries(self, row, scratch_cache: Path, raw_dir: Path) -> str | None:
        dest = raw_dir / f"{row.volume_id}.nii.gz"
        for attempt in range(self.retries + 1):
            try:
                download_volume(self.hf_hub_download, self.cfg.ingest.hf_repo, row.repo_path, scratch_cache, dest)
                return None
            except Exception as exc:  # noqa: BLE001 - network/auth/404: report it per volume
                if attempt == self.retries:
                    return f"{type(exc).__name__}: {exc}"
                _sleep(2 * (attempt + 1))
        return None

    def fetch(self, chunk_id: str, raw_dir: Path, *, cache_fresh: Callable[[str], bool]) -> FetchReport:
        rows = self._chunk_rows(chunk_id)
        needed = [r for r in rows.itertuples() if not cache_fresh(r.volume_id)]
        report = FetchReport()
        if not needed:
            return report
        scratch_cache = raw_dir / ".hf_cache"
        with ThreadPoolExecutor(max_workers=self.cfg.ingest.fetch_workers) as ex:
            errors = list(ex.map(lambda r: self._download_with_retries(r, scratch_cache, raw_dir), needed))
        shutil.rmtree(scratch_cache, ignore_errors=True)
        for r, err in zip(needed, errors):
            if err:
                report.failed[r.volume_id] = err
        if len(report.failed) == len(needed):
            first = next(iter(report.failed.values()))
            raise FetchError(f"every download in chunk {chunk_id} failed (network or Hugging Face login?): {first}")
        return report

    def build_rows(self, chunk_id: str, raw_dir: Path, *, patient_id_source: str = "path") -> pd.DataFrame:
        ids = self._chunk_rows(chunk_id)["volume_id"].tolist()
        return build_manifest_ctrate(ids, self.metadata)
