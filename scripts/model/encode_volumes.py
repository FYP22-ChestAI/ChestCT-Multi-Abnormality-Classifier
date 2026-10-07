"""Stage 2: encode every selected volume of a run into the embedding store.

    python scripts/model/encode_volumes.py --run train-sharp
    python scripts/model/encode_volumes.py --run train-sharp --dry-run            # what would be encoded, disk estimate
    python scripts/model/encode_volumes.py --run soft-subset --kernel-classes soft
    python scripts/model/encode_volumes.py --run train-sharp --encoder dinov2_vitl14   # another backbone = another store

Reads the run's manifest (data/runs/<source>/<run>/manifest.csv) and the shared HU cache (never
modified), and writes data/embeddings/<encoder>-<fingerprint>/{volume_id}.npy: (n_slices, D) float16,
one row per cache slice. Volumes already in the store are skipped, so the script can be stopped and
re-run at any time, and a second run only encodes what is missing.

Defaults come from configs/model/encode.yaml and the encoder file configs/model/encoders/<name>.yaml;
any default can be overridden here. Exit code 1 if any volume failed or the run stopped early.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ct_preprocessing.cache_record import read_cache_fingerprint
from ct_preprocessing.cli import friendly_errors, pick, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.ingest.engine import free_gb

from ct_model.config import DEFAULT_ENCODE_CONFIG, load_encoder_config, load_stage2_config
from ct_model.data.volumes import load_run_manifest, missing_cache_files, select_volumes
from ct_model.embeddings.store import open_store, read_store_record, store_dir
from ct_model.utils.log import stream_logs


_BYTES = {"float16": 2, "float32": 4}


def _size(n_bytes: float) -> str:
    return f"{n_bytes / 1e9:.2f} GB" if n_bytes >= 1e9 else f"{n_bytes / 1e6:.1f} MB"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_ENCODE_CONFIG, help=f"stage-2 settings (default: {DEFAULT_ENCODE_CONFIG})")
    ap.add_argument("--encoder", help="encoder config name in configs/model/encoders/ or a path to one")
    ap.add_argument("--source")
    ap.add_argument("--run", help="run name (default: the only run of the source)")
    ap.add_argument("--splits", nargs="+", choices=["train", "val", "test"])
    ap.add_argument("--kernel-classes", nargs="+", help="e.g. sharp soft (default: every kernel class)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--amp-dtype", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--slice-batch-size", type=int)
    ap.add_argument("--num-workers", type=int)
    ap.add_argument("--min-free-gb", type=float)
    ap.add_argument("--limit", type=int, help="encode at most this many volumes (smoke tests)")
    ap.add_argument("--dry-run", action="store_true", help="only report what would be encoded; nothing is written")
    return ap


@friendly_errors
def main() -> int:
    stream_logs()  # progress lines reach | tee / log files as they happen
    args = build_parser().parse_args()
    cfg = load_stage2_config(args.config)
    sel, enc = cfg.selection, cfg.encode
    data_cfg = load_config(cfg.data_config)
    enc_cfg = load_encoder_config(pick(args.encoder, cfg.encoder))

    settings = {
        "encoder": enc_cfg.name, "source": pick(args.source, sel.source), "run": pick(args.run, sel.run),
        "splits": tuple(pick(args.splits, sel.splits)), "qc_passed_only": sel.qc_passed_only,
        "kernel_classes": tuple(args.kernel_classes) if args.kernel_classes else sel.kernel_classes,
        "device": pick(args.device, enc.device), "amp_dtype": pick(args.amp_dtype, enc.amp_dtype),
        "slice_batch_size": pick(args.slice_batch_size, enc.slice_batch_size),
        "num_workers": pick(args.num_workers, enc.num_workers),
        "min_free_gb": pick(args.min_free_gb, enc.min_free_gb), "limit": pick(args.limit, enc.limit),
        "embeddings_dir": enc.embeddings_dir,
    }
    run, manifest = load_run_manifest(data_cfg, settings["source"], settings["run"])
    settings["run"] = run.name
    show_settings("encode_volumes", settings)

    records = select_volumes(
        manifest, data_cfg.paths.cache_dir, splits=settings["splits"],
        qc_passed_only=settings["qc_passed_only"], kernel_classes=settings["kernel_classes"],
    )
    if settings["limit"]:
        records = records[: settings["limit"]]
    if not records:
        raise ValueError(f"no volumes of run {run.name!r} match the selection -- check splits / kernel_classes")
    missing = missing_cache_files(records)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected volume(s) are in the manifest but not in {data_cfg.paths.cache_dir} "
            f"(e.g. {missing[:5]}) -- the cache is the source of truth; re-run ingest or fix the manifest"
        )

    folder = store_dir(settings["embeddings_dir"], enc_cfg)
    hu_fp = read_cache_fingerprint(data_cfg.paths.cache_dir)
    n_slices = sum(r.n_slices for r in records)
    dim = enc_cfg.embed_dim
    est = _size(n_slices * dim * _BYTES[enc_cfg.store_dtype]) if dim else "unknown (no embed_dim in the encoder config)"
    by_split = {s: sum(r.split == s for r in records) for s in settings["splits"]}
    print(f"selected {len(records)} volumes ({n_slices} slices; {by_split}) -> {folder}  [their embeddings: ~{est}]")
    print(f"encoder fingerprint {enc_cfg.fingerprint()}, HU cache fingerprint {hu_fp}")

    done = 0
    if read_store_record(folder):  # also checks the existing store still matches the HU cache
        existing = open_store(settings["embeddings_dir"], enc_cfg, hu_cache_fingerprint=hu_fp, create=False)
        done = sum(existing.has(r.volume_id, r.n_slices) for r in records)
    if args.dry_run:
        print(f"dry run: {done} already encoded, {len(records) - done} to encode")
        return 0
    if done == len(records):  # nothing to do: do not spend minutes loading the model
        print(f"done: all {done} selected volumes are already in the store")
        return 0

    free = free_gb(Path(settings["embeddings_dir"]))
    if settings["min_free_gb"] and free < settings["min_free_gb"]:  # before minutes of model loading, not after
        raise RuntimeError(f"only {free:.1f} GB free on {settings['embeddings_dir']}, below min_free_gb "
                           f"{settings['min_free_gb']} -- free disk, or lower --min-free-gb deliberately")

    from ct_model.embeddings.extract import encode_volumes
    from ct_model.encoders import build_encoder
    from ct_model.utils.device import describe_device, resolve_amp_dtype, resolve_device

    device = resolve_device(settings["device"])
    amp_dtype = resolve_amp_dtype(settings["amp_dtype"], device)
    print(f"device {describe_device(device)}, autocast {amp_dtype or 'off (float32)'}", flush=True)
    encoder = build_encoder(enc_cfg)
    store = open_store(
        settings["embeddings_dir"], enc_cfg, hu_cache_fingerprint=hu_fp,
        embed_dim=encoder.embed_dim, provenance=encoder.provenance(),
    )
    summary = encode_volumes(
        records, encoder, store, device=device, amp_dtype=amp_dtype,
        slice_batch_size=settings["slice_batch_size"], num_workers=settings["num_workers"],
        min_free_gb=settings["min_free_gb"],
    )
    store.write_index()

    vram = f", peak VRAM {summary.peak_vram_gb:.1f} GiB" if summary.peak_vram_gb is not None else ""
    print(
        f"done: {summary.encoded} encoded, {summary.skipped} already in the store, {len(summary.failed)} failed "
        f"({summary.slices_per_second:.0f} slices/s, slice_batch_size {summary.slice_batch_size}{vram}); "
        f"store now {len(store.volume_ids())} volumes, {_size(store.size_gb() * 1e9)}"
    )
    if summary.failed:
        report = Path(store.dir) / f"failed_{run.source}_{run.name}.csv"
        pd.DataFrame(list(summary.failed.items()), columns=["volume_id", "error"]).to_csv(report, index=False)
        print(f"failed volumes listed in {report}")
    if summary.stopped:
        print(f"stopped early: {summary.stopped} -- free disk and run again; finished volumes are kept")
    return 1 if summary.failed or summary.stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
