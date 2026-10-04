"""Build the right fetcher for a configured source."""
from __future__ import annotations

from typing import Callable

from ..config import DataConfig
from ..manifest import CTRATE_BUILDER
from . import state
from .archives import ArchiveFetcher, make_backend
from .base import Fetcher
from .ctrate import CtrateFetcher, load_metadata
from .worklist import read_worklist


def get_hf_download() -> Callable:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise SystemExit("pip install huggingface_hub first, then run: huggingface-cli login") from exc
    return hf_hub_download


def build_fetcher(
    source_name: str,
    cfg: DataConfig,
    *,
    hf_hub_download: Callable | None = None,
    remote: str | None = None,
) -> Fetcher:
    """CT-RATE sources read their worklist + metadata and download from Hugging
    Face; folder sources read archives from ``remote`` (default: the source's
    configured ``ingest.drive_remote``)."""
    source_cfg = cfg.source(source_name)
    if source_cfg.manifest_builder == CTRATE_BUILDER:
        worklist = read_worklist(state.worklist_path(cfg.paths, source_name))
        return CtrateFetcher(worklist, load_metadata(cfg.paths), source_cfg, hf_hub_download or get_hf_download())
    remote = remote or source_cfg.ingest.drive_remote
    if not remote:
        raise ValueError(
            f"sources.{source_name}.ingest.drive_remote is not set -- set it in the config or pass --drive-remote"
        )
    return ArchiveFetcher(make_backend(remote), source_cfg)
