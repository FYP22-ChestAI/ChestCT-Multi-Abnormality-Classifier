"""Compare experiments: group the ledgers by configuration and average over seeds.

    python scripts/model/summarize_experiments.py                    # val results of every config
    python scripts/model/summarize_experiments.py --split test       # test results (evaluations.csv)
    python scripts/model/summarize_experiments.py --name abmil_dale2s

Rows with the same config hash are the same experiment with different seeds: mean, std and the number
of seeds are reported per metric. Writes outputs/experiments/summary_<split>.csv as well.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ct_preprocessing.cli import friendly_errors

from ct_model.training.records import EVAL_LEDGER, TRAIN_LEDGER
from ct_model.utils.log import stream_logs


_KEYS = ["name", "config_hash", "aggregator", "loss", "lr"]


def summarize(output_dir: str | Path, split: str = "val", name: str | None = None) -> pd.DataFrame:
    output_dir = Path(output_dir)
    train_path = output_dir / TRAIN_LEDGER
    if not train_path.is_file():
        raise FileNotFoundError(f"no {train_path} yet -- train something with scripts/model/train_mil.py first")
    runs = pd.read_csv(train_path)
    if split == "val":
        frame = runs
        metrics = ["val_macro_auroc", "val_macro_auprc", "val_micro_auroc", "val_macro_f1", "best_epoch"]
    else:
        eval_path = output_dir / EVAL_LEDGER
        if not eval_path.is_file():
            raise FileNotFoundError(f"no {eval_path} yet -- run scripts/model/evaluate_mil.py first")
        evals = pd.read_csv(eval_path)
        evals = evals[evals.eval_split == split]
        frame = evals.merge(runs[["run_id", "aggregator", "loss", "lr"]], on="run_id", how="left")
        metrics = [c for c in frame.columns if c.startswith(("macro_auroc", "macro_auprc", "micro_auroc", "macro_f1"))
                   and "ci_" not in c]
    if name:
        frame = frame[frame.name == name]
    if frame.empty:
        raise ValueError("no matching rows in the ledger")
    keys = [k for k in _KEYS if k in frame.columns]
    grouped = frame.groupby(keys, dropna=False)
    out = grouped[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out.insert(0, "seeds", grouped.size())
    return out.reset_index().sort_values(f"{metrics[0]}_mean", ascending=False)


@friendly_errors
def main() -> int:
    stream_logs()  # progress lines reach | tee / log files as they happen
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default="outputs/experiments")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--name", help="only this experiment name")
    args = ap.parse_args()
    table = summarize(args.output_dir, args.split, args.name)
    path = Path(args.output_dir) / f"summary_{args.split}.csv"
    table.to_csv(path, index=False, lineterminator="\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
