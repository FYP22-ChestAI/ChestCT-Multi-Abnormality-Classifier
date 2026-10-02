"""What every data source has to provide to the ingest engine.

Only HOW raw data reaches the scratch folder differs between sources
(Hugging Face files for CT-RATE, uploaded archives for NHRD); everything after
that -- manifest rows, preprocessing, cleanup, resuming -- is shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import pandas as pd


class FetchError(RuntimeError):
    """A chunk could not be fetched or unpacked (network, auth, corrupt archive)."""


@dataclass
class FetchReport:
    # volume_id -> reason, for individual volumes that could not be fetched while
    # the rest of the chunk was. They are recorded as failed volumes, not as a failed chunk.
    failed: dict[str, str] = field(default_factory=dict)


class Fetcher(Protocol):
    def chunk_ids(self) -> list[str]:
        """Every chunk this source currently offers, in processing order."""

    def fetch(self, chunk_id: str, raw_dir: Path, *, cache_fresh: Callable[[str], bool]) -> FetchReport:
        """Put the chunk's raw files under ``raw_dir``. ``cache_fresh(volume_id)``
        says whether a volume is already cached, so a resumed chunk need not
        fetch it again. Raise FetchError if the chunk as a whole cannot be fetched."""

    def build_rows(self, chunk_id: str, raw_dir: Path, *, patient_id_source: str) -> pd.DataFrame:
        """One manifest row per volume in the chunk, with no split."""
