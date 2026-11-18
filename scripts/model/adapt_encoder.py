"""Phase 2: adapt the slice encoder with LoRA, end to end from the HU cache, with sharp / soft consistency.

    python scripts/model/adapt_encoder.py --adapt-config lora_dale2s_kernel
    python scripts/model/adapt_encoder.py --adapt-config lora_dale2s_kernel --init-from outputs/experiments/<name>/<hash8>/seed0

Trains LoRA adapters in the last blocks of the encoder (plus the aggregator and head, initialised from a
phase-1 run) on k sampled slices per volume. Besides the volume labels (BCE), two label-free terms target the
kernel gap: the aligned slices of a scan's sharp and soft reconstructions should embed alike, and the two
should predict alike (one-kernel scans get a blurred / sharpened copy instead). The best epoch on val is kept.

Writes outputs/adapters/<name>/<hash8>/seed<k>/ (adapter.safetensors, encoder.yaml, experiment.yaml, metrics.csv)
and prints the next steps: encode the adapted encoder's store, then train / evaluate the MIL head on it like any
phase-1 experiment, so the clinical report compares it with phase 1 on the same yardstick. Running the same
command again resumes an interrupted run. Settings: configs/model/adapt/<name>.yaml (see ct_model.adapt).
"""
from __future__ import annotations

import argparse

from ct_preprocessing.cli import friendly_errors, show_settings

from ct_model.utils.log import stream_logs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapt-config", required=True, help="a file in configs/model/adapt/ (without .yaml), or a path")
    ap.add_argument("--init-from", help="a finished phase-1 run to start the aggregator + head from (default: the config's)")
    ap.add_argument("--run", help="data run (default: data.run of the config; 'a+b' joins runs)")
    ap.add_argument("--seed", type=int, help="default: the config's seed")
    ap.add_argument("--epochs", type=int, help="default: optim.epochs of the config")
    ap.add_argument("--scans-per-epoch", type=int, help="cap the train scans per epoch (default: all)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--workers", type=int, default=4, help="DataLoader processes reading the HU cache")
    return ap


@friendly_errors
def main() -> int:
    stream_logs()
    args = build_parser().parse_args()

    from ct_model.adapt.config import load_adapt_config, parse_adapt_config
    from ct_model.adapt.trainer import train_adapter

    cfg = load_adapt_config(args.adapt_config)
    raw = cfg.to_dict()
    for key, value in (("init_from", args.init_from), ("seed", args.seed), ("device", args.device)):
        if value is not None:
            raw[key] = value
    if args.run:
        raw["data"]["run"] = args.run
    if args.epochs:
        raw["optim"]["epochs"] = args.epochs
    if args.scans_per_epoch:
        raw["optim"]["scans_per_epoch"] = args.scans_per_epoch
    cfg = parse_adapt_config(raw, origin=f"{args.adapt_config} + arguments")
    show_settings("adapt_encoder", {"name": cfg.name, "encoder": cfg.encoder, "init_from": cfg.init_from,
                                    "run": cfg.data.run, "seed": cfg.seed, "epochs": cfg.optim.epochs,
                                    "lora": cfg.lora, "workers": args.workers})
    train_adapter(cfg, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
