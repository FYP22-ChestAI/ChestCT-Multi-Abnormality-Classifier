"""Build the right fetcher for a configured source."""
from __future__ import annotations

from typing import Callable

from ..config import DataConfig
from ..manifest import CTRATE_BUILDER
from ..runs import Run
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


def archive_remote(cfg: DataConfig, source_name: str, remote: str | None = None) -> str:
    remote = remote or cfg.source(source_name).ingest.drive_remote
    if not remote:
        raise ValueError(
            f"sources.{source_name}.ingest.drive_remote is not set -- set it in the config or pass --drive-remote"
        )
    return remote


def build_fetcher(
    run: Run,
    cfg: DataConfig,
    *,
    hf_hub_download: Callable | None = None,
    remote: str | None = None,
) -> Fetcher:
    """CT-RATE sources read the run's worklist + the metadata and download from Hugging
    Face; folder sources read archives from ``remote`` (default: the source's configured
    ``ingest.drive_remote``)."""
    source_cfg = cfg.source(run.source)
    if source_cfg.manifest_builder == CTRATE_BUILDER:
        worklist = read_worklist(run.worklist_path)
        return CtrateFetcher(worklist, load_metadata(cfg.paths), source_cfg, hf_hub_download or get_hf_download())
    return ArchiveFetcher(make_backend(archive_remote(cfg, run.source, remote)), source_cfg)
