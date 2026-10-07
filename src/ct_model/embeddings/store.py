"""The embedding store: a DERIVED cache of per-slice embeddings, one folder per encoder config.

    data/embeddings/<encoder name>-<fingerprint8>/
        .embedding_record.json     what made these embeddings (encoder config + fingerprint, weights commit,
                                   HU-cache fingerprint, dim, dtype)
        {volume_id}.npy            (n_slices, embed_dim), float16 by default; row i = cache slice i (head -> foot)
        index.csv                  volume_id, n_slices, embed_dim (rewritten after every encode run)

The HU cache (data/cache) stays the source of truth. This folder can be deleted and rebuilt at any
time, and it is SHARED by every run, like the HU cache: a second run (e.g. a soft-kernel subset) only
encodes the volumes the store does not hold yet.

It refuses to mix embeddings: a store whose record says it was built from another HU cache (the
cache's .cache_fingerprint.json), other encoder settings or other weights raises StoreMismatch.

float16 is lossless here. The encoder runs under bfloat16 autocast on the GPU, and every bfloat16 value
in fp16's normal range is exactly representable in float16 (10 mantissa bits vs 7). Values outside
that range (|x| >= 65504, or NaN / inf) are refused rather than stored as inf.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ct_preprocessing.ingest.state import write_atomic

from ..config import EncoderConfig

STORE_RECORD = ".embedding_record.json"
INDEX_FILE = "index.csv"
_FP16_MAX = float(np.finfo(np.float16).max)


class StoreMismatch(RuntimeError):
    """The store folder holds embeddings made from another HU cache or with other encoder settings."""


@dataclass(frozen=True)
class EmbeddingStore:
    dir: Path
    embed_dim: int
    dtype: str  # "float16" | "float32"

    def path(self, volume_id: str) -> Path:
        return self.dir / f"{volume_id}.npy"

    def has(self, volume_id: str, n_slices: int | None = None) -> bool:
        """True if the volume's embeddings are on disk and have the right shape (only the .npy header is read)."""
        path = self.path(volume_id)
        if not path.is_file():
            return False
        try:
            arr = np.load(path, mmap_mode="r")
        except (OSError, ValueError):
            return False
        ok = arr.ndim == 2 and arr.shape[1] == self.embed_dim and arr.dtype == np.dtype(self.dtype)
        return ok and (n_slices is None or arr.shape[0] == n_slices)

    def load(self, volume_id: str, mmap: bool = True) -> np.ndarray:
        return np.load(self.path(volume_id), mmap_mode="r" if mmap else None)

    def write(self, volume_id: str, embeddings: np.ndarray) -> Path:
        """Validate and save one volume's (n_slices, embed_dim) embeddings, atomically."""
        emb = np.asarray(embeddings)
        if emb.ndim != 2 or emb.shape[1] != self.embed_dim or emb.shape[0] < 1:
            raise ValueError(f"{volume_id}: expected (n_slices, {self.embed_dim}) embeddings, got {emb.shape}")
        if not np.isfinite(emb).all():
            raise ValueError(f"{volume_id}: embeddings contain NaN or inf -- not stored")
        if self.dtype == "float16" and np.abs(emb).max() >= _FP16_MAX:
            raise ValueError(f"{volume_id}: embeddings exceed the float16 range -- use store_dtype float32")
        path = self.path(volume_id)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, emb.astype(self.dtype, copy=False))
        os.replace(tmp, path)
        return path

    def volume_ids(self) -> list[str]:
        return sorted(p.name[: -len(".npy")] for p in self.dir.glob("*.npy"))

    def write_index(self) -> Path:
        rows = []
        for vid in self.volume_ids():
            try:
                rows.append({"volume_id": vid, "n_slices": int(self.load(vid).shape[0]), "embed_dim": self.embed_dim})
            except (OSError, ValueError):
                continue
        path = self.dir / INDEX_FILE
        write_atomic(path, pd.DataFrame(rows, columns=["volume_id", "n_slices", "embed_dim"]).to_csv(index=False, lineterminator="\n"))
        return path

    def size_gb(self) -> float:
        return sum(p.stat().st_size for p in self.dir.glob("*.npy")) / 1e9

    def read_record(self) -> dict:
        return read_store_record(self.dir)


def store_dir(embeddings_dir: str | Path, enc_cfg: EncoderConfig) -> Path:
    return Path(embeddings_dir) / enc_cfg.store_name


def read_store_record(folder: str | Path) -> dict:
    try:
        return json.loads((Path(folder) / STORE_RECORD).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _differs(recorded, now) -> bool:
    """Only two known values can disagree; an unknown (None) on either side is not a mismatch."""
    return recorded is not None and now is not None and recorded != now


def open_store(
    embeddings_dir: str | Path,
    enc_cfg: EncoderConfig,
    *,
    hu_cache_fingerprint: str | None,
    embed_dim: int | None = None,
    provenance: dict | None = None,
    create: bool = True,
) -> EmbeddingStore:
    """The store for ``enc_cfg``. Creates it (``create``) or checks it still matches what is asked for.

    Stage 2 passes ``embed_dim`` and ``provenance`` from the built encoder; stages 3 + 4 open an existing
    store with ``create=False`` and only the config and the HU-cache fingerprint.
    """
    folder = store_dir(embeddings_dir, enc_cfg)
    record = read_store_record(folder)
    embed_dim = embed_dim if embed_dim is not None else enc_cfg.embed_dim
    if record:
        problems = []
        if record.get("encoder_fingerprint") != enc_cfg.fingerprint():
            problems.append(f"encoder fingerprint {record.get('encoder_fingerprint')} vs {enc_cfg.fingerprint()}")
        if _differs(record.get("hu_cache_fingerprint"), hu_cache_fingerprint):
            problems.append(f"HU cache fingerprint {record.get('hu_cache_fingerprint')} vs {hu_cache_fingerprint} "
                            "(the cache was rebuilt with other preprocessing settings)")
        if _differs(record.get("embed_dim"), embed_dim):
            problems.append(f"embed_dim {record.get('embed_dim')} vs {embed_dim}")
        recorded_commit = (record.get("provenance") or {}).get("weights_commit")
        if _differs(recorded_commit, (provenance or {}).get("weights_commit")):
            problems.append(f"weights commit {recorded_commit} vs {provenance['weights_commit']}")
        if problems:
            raise StoreMismatch(
                f"the embedding store {folder} was made differently: {'; '.join(problems)}. "
                "Delete that folder to rebuild it (the HU cache is untouched), or restore the old settings."
            )
        return EmbeddingStore(folder, int(record["embed_dim"]), record["store_dtype"])
    if not create:
        raise FileNotFoundError(f"no embedding store {folder} -- run scripts/model/encode_volumes.py --encoder {enc_cfg.name} first")
    if embed_dim is None:
        raise ValueError("embed_dim is needed to create a store (from the built encoder or the encoder config)")
    folder.mkdir(parents=True, exist_ok=True)
    write_atomic(folder / STORE_RECORD, json.dumps({
        "store": enc_cfg.store_name,
        "encoder_name": enc_cfg.name,
        "encoder_fingerprint": enc_cfg.fingerprint(),
        "encoder_config": asdict(enc_cfg),
        "embed_dim": int(embed_dim),
        "store_dtype": enc_cfg.store_dtype,
        "hu_cache_fingerprint": hu_cache_fingerprint,
        "provenance": provenance or {},
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2, default=list))
    return EmbeddingStore(folder, int(embed_dim), enc_cfg.store_dtype)
