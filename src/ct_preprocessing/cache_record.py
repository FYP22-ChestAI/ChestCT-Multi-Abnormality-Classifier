"""The cache remembers which preprocessing settings built it.

Every run writes into one shared cache, and a cached volume is only valid for the
settings (fingerprint) that produced it. If the settings change and a second run
ingests into the same folder, the files an EARLIER run's manifest points to would be
silently overwritten with differently prepared volumes. So the cache folder carries a
small record of its fingerprint, and ingest refuses to write into it with other
settings unless that is asked for explicitly. Inference compares against the same record.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .ingest.state import write_atomic
from .preprocess import config_fingerprint
from .preprocess_config import PreprocessConfig

CACHE_RECORD = ".cache_fingerprint.json"
_SAMPLE = 50  # sidecars read to adopt a cache that has no record yet


class CacheMismatch(RuntimeError):
    """The cache was built with other preprocessing settings than the ones in use."""


def record_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / CACHE_RECORD


def read_cache_fingerprint(cache_dir: str | Path) -> str | None:
    try:
        return json.loads(record_path(cache_dir).read_text(encoding="utf-8")).get("fingerprint")
    except (OSError, json.JSONDecodeError):
        return None


def _write(cache_dir: Path, cfg: PreprocessConfig) -> None:
    write_atomic(
        record_path(cache_dir),
        json.dumps({"fingerprint": config_fingerprint(cfg), "preprocess_config": asdict(cfg)}, indent=2, default=str),
    )


def _sidecar_fingerprints(cache_dir: Path) -> set[str]:
    found: set[str] = set()
    for i, sidecar in enumerate(sorted(cache_dir.glob("*.meta.json"))):
        if i >= _SAMPLE:
            break
        try:
            found.add(json.loads(sidecar.read_text(encoding="utf-8")).get("fingerprint", ""))
        except (OSError, json.JSONDecodeError):
            continue
    return found


def ensure_cache_matches(cache_dir: str | Path, cfg: PreprocessConfig, *, allow_new_settings: bool = False) -> None:
    """Make sure ingesting with ``cfg`` into ``cache_dir`` cannot overwrite volumes built with other settings.

    * No record and an empty cache: write the record.
    * No record but a cache from before records existed: adopt it if its volumes carry ``cfg``'s fingerprint.
    * A record that differs: raise CacheMismatch, unless ``allow_new_settings`` -- then the record is replaced
      and every cached volume built with the old settings counts as stale (it is fetched again when needed).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    now = config_fingerprint(cfg)
    recorded = read_cache_fingerprint(cache_dir)
    if recorded is None:
        existing = _sidecar_fingerprints(cache_dir)
        if existing and existing != {now} and not allow_new_settings:
            raise CacheMismatch(
                f"the cache in {cache_dir} holds volumes built with other preprocessing settings "
                f"(fingerprint(s) {sorted(existing)}, now {now}). Ingesting would overwrite files that existing "
                "runs point to. Restore the settings in the config, use another cache_dir, or pass "
                "--allow-new-settings to replace them (old volumes are then stale)."
            )
        _write(cache_dir, cfg)
    elif recorded != now:
        if not allow_new_settings:
            raise CacheMismatch(
                f"the cache in {cache_dir} was built with other preprocessing settings (fingerprint {recorded}, now {now}). "
                "Ingesting would overwrite volumes that existing runs point to. Restore the settings in the config, "
                "use another cache_dir, or pass --allow-new-settings to replace them (old volumes are then stale)."
            )
        _write(cache_dir, cfg)
