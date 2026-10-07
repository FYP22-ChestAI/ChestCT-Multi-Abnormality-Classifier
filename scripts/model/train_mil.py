"""Stages 3 + 4, phase 1: train an aggregator + head on the frozen slice embeddings (train + val only).

    python scripts/model/train_mil.py --experiment abmil_dale2s
    python scripts/model/train_mil.py --experiment abmil_dale2s --seed 1
    python scripts/model/train_mil.py --experiment abmil_dale2s --loss asl --lr 1e-4

Reads configs/model/experiments/<experiment>.yaml; the arguments below override it for this run. The
run's embeddings must already exist (scripts/model/encode_volumes.py). Results go to
outputs/experiments/<name>/<config hash>/seed<k>/ and one row is added to outputs/experiments/index.csv
(see docs/model/README.md). The test split is never read here: run evaluate_mil.py on the finished run.

Running the same command again resumes an interrupted run; a finished one is refused.
"""
from __future__ import annotations

import argparse

from ct_preprocessing.cli import friendly_errors, show_settings

from ct_model.training.config import load_experiment, override
from ct_model.utils.log import stream_logs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="a file name in configs/model/experiments/ (without .yaml), or a path")
    ap.add_argument("--run", help="data run (default: data.run of the experiment)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--loss", choices=["bce", "weighted_bce", "asl"], help="loss type (its parameters keep their defaults)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--output-dir", help="default: output_dir of the experiment (outputs/experiments)")
    return ap


@friendly_errors
def main() -> int:
    stream_logs()  # progress lines reach | tee / log files as they happen
    args = build_parser().parse_args()
    cfg = override(
        load_experiment(args.experiment),
        **{"data.run": args.run, "seed": args.seed, "loss": {"type": args.loss} if args.loss else None,
           "optim.lr": args.lr, "optim.weight_decay": args.weight_decay, "optim.batch_size": args.batch_size,
           "optim.max_epochs": args.max_epochs, "optim.patience": args.patience, "device": args.device,
           "output_dir": args.output_dir},
    )
    show_settings("train_mil", {
        "experiment": cfg.name, "run": cfg.data.run, "seed": cfg.seed, "encoder": cfg.encoder,
        "aggregator": cfg.aggregator, "head": cfg.head, "loss": cfg.loss, "lr": cfg.optim.lr,
        "weight_decay": cfg.optim.weight_decay, "batch_size": cfg.optim.batch_size,
        "max_epochs": cfg.optim.max_epochs, "patience": cfg.optim.patience, "monitor": cfg.optim.monitor,
        "device": cfg.device, "output_dir": cfg.output_dir,
    })
    from ct_model.training.trainer import train_experiment

    result = train_experiment(cfg)
    s = result.val_summary
    print(f"val (best epoch {result.best_epoch}): macro AUROC {s['macro_auroc']:.4f}, macro AUPRC {s['macro_auprc']:.4f}, "
          f"micro AUROC {s['micro_auroc']:.4f}, macro F1 {s['macro_f1']:.4f} at val-chosen thresholds")
    print(f"next: python scripts/model/evaluate_mil.py --experiment-dir {result.dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
