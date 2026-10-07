"""Smoke test for a backbone, before a long encode run: nothing is written.

    python scripts/model/check_encoder.py --run train-sharp
    python scripts/model/check_encoder.py --run train-sharp --encoder dale_ct_2s --volume-id train_1156_a_1 --device cpu

Builds the encoder (downloading its weights once into the Hugging Face cache), takes one cached volume
of the run (the first selected one, or --volume-id), and reports:
  * the encoder: fingerprint, parameter count, where its weights came from;
  * its input: shape and value range after the backbone's own HU transform;
  * its output: shape, finite, norms, and how similar neighbouring slices are (should be high, < 1);
  * speed and peak GPU memory, and the projected time for the whole selection.
Use it on the server to pick slice_batch_size (configs/model/encode.yaml) for the GPU.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ct_preprocessing.cli import friendly_errors, pick, show_settings
from ct_preprocessing.config import load_config

from ct_model.config import DEFAULT_ENCODE_CONFIG, load_encoder_config, load_stage2_config
from ct_model.data.slices import SliceSampler
from ct_model.data.volumes import load_run_manifest, select_volumes
from ct_model.utils.log import stream_logs


@friendly_errors
def main() -> int:
    stream_logs()  # progress lines reach | tee / log files as they happen
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_ENCODE_CONFIG)
    ap.add_argument("--encoder")
    ap.add_argument("--source")
    ap.add_argument("--run")
    ap.add_argument("--volume-id", help="default: the first selected volume of the run")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--amp-dtype", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--slice-batch-size", type=int)
    ap.add_argument("--max-slices", type=int, default=64, help="slices to encode, evenly spread (0 = the whole volume)")
    args = ap.parse_args()

    import torch

    from ct_model.embeddings.extract import encode_slices
    from ct_model.encoders import build_encoder
    from ct_model.utils.device import describe_device, resolve_amp_dtype, resolve_device

    cfg = load_stage2_config(args.config)
    data_cfg = load_config(cfg.data_config)
    enc_cfg = load_encoder_config(pick(args.encoder, cfg.encoder))
    run, manifest = load_run_manifest(data_cfg, pick(args.source, cfg.selection.source), pick(args.run, cfg.selection.run))
    records = select_volumes(manifest, data_cfg.paths.cache_dir, splits=cfg.selection.splits,
                             qc_passed_only=cfg.selection.qc_passed_only, kernel_classes=cfg.selection.kernel_classes)
    if args.volume_id:
        records_by_id = {r.volume_id: r for r in records}
        if args.volume_id not in records_by_id:
            raise KeyError(f"{args.volume_id} is not among the selected volumes of run {run.name!r}")
        rec = records_by_id[args.volume_id]
    elif records:
        rec = records[0]
    else:
        raise ValueError(f"run {run.name!r} has no selected volumes")

    device = resolve_device(pick(args.device, cfg.encode.device))
    amp_dtype = resolve_amp_dtype(pick(args.amp_dtype, cfg.encode.amp_dtype), device)
    batch = pick(args.slice_batch_size, cfg.encode.slice_batch_size)
    show_settings("check_encoder", {"encoder": enc_cfg.name, "run": run.name, "volume": rec.volume_id,
                                    "device": describe_device(device), "autocast": amp_dtype or "off",
                                    "slice_batch_size": batch, "max_slices": args.max_slices})

    t0 = time.perf_counter()
    encoder = build_encoder(enc_cfg).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"encoder {enc_cfg.store_name}: {n_params / 1e6:.1f} M parameters, embed_dim {encoder.embed_dim}, "
          f"built in {time.perf_counter() - t0:.1f} s")
    for k, v in encoder.provenance().items():
        print(f"  {k}: {v}")

    volume = np.load(rec.npy_path, mmap_mode="r")
    sampler = SliceSampler("all") if args.max_slices == 0 else SliceSampler("uniform_k", k=args.max_slices)
    idx = sampler(volume.shape[0])
    hu = torch.from_numpy(np.ascontiguousarray(volume[idx]))
    x = encoder.input_transform(hu[:4].to(device))
    print(f"volume {rec.volume_id}: cache {volume.shape} {volume.dtype}, HU [{int(hu.min())}, {int(hu.max())}]; "
          f"encoding {len(idx)} slices")
    print(f"  model input {tuple(x.shape)}: min {x.min():.3f} max {x.max():.3f} mean {x.mean():.3f} std {x.std():.3f}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        encode_slices(encoder, hu[:2], device=device, amp_dtype=amp_dtype, slice_batch_size=2)  # warm-up
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    emb, used = encode_slices(encoder, hu, device=device, amp_dtype=amp_dtype, slice_batch_size=batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    rate = len(idx) / seconds

    norms = np.linalg.norm(emb, axis=1)
    unit = emb / np.maximum(norms[:, None], 1e-12)
    neighbour = (unit[1:] * unit[:-1]).sum(1) if len(unit) > 1 else np.array([np.nan])
    print(f"  output {emb.shape} (finite: {bool(np.isfinite(emb).all())}), |x| max {np.abs(emb).max():.2f}, "
          f"norm mean {norms.mean():.2f}, neighbour-slice cosine {np.nanmean(neighbour):.3f}")
    vram = f", peak VRAM {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB" if device.type == "cuda" else ""
    print(f"  {rate:.1f} slices/s at slice_batch_size {used}{vram}")
    total = sum(r.n_slices for r in records)
    print(f"projected: {len(records)} selected volumes, {total} slices -> ~{total / rate / 3600:.1f} h at this speed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
