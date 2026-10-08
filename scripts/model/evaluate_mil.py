"""Evaluate a finished training run -- normally on test, once, after every choice was made on val.

    python scripts/model/evaluate_mil.py --experiment-dir outputs/experiments/abmil_dale2s/1a2b3c4d/seed0
    python scripts/model/evaluate_mil.py --experiment-dir ... --run soft-subset          # another run, no retraining

Uses the best checkpoint and the per-label thresholds chosen on val at training time. Writes
<run dir>/eval/<source>__<run>__<split>/ (predictions, per-label and summary metrics: overall, per
kernel class, per scanner, with patient-level bootstrap 95 % CIs) and adds a row to
outputs/experiments/evaluations.csv. An evaluation that already exists is refused (--overwrite redoes it).
"""
from __future__ import annotations

import argparse

from ct_preprocessing.cli import friendly_errors, show_settings

from ct_model.utils.log import stream_logs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-dir", required=True, help="a finished run: outputs/experiments/<name>/<hash>/seed<k>")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--source", help="default: the experiment's source")
    ap.add_argument("--run", help="default: the run the experiment was trained on")
    ap.add_argument("--kernel-classes", nargs="+", help="only these kernel classes (default: all)")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for the CIs (0 = none)")
    ap.add_argument("--min-group", type=int, default=100, help="groups smaller than this are flagged too_small, no CI")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--overwrite", action="store_true")
    return ap


@friendly_errors
def main() -> int:
    stream_logs()  # progress lines reach | tee / log files as they happen
    args = build_parser().parse_args()
    show_settings("evaluate_mil", {k: v for k, v in vars(args).items()})

    from ct_model.training.evaluate import evaluate_experiment

    result = evaluate_experiment(
        args.experiment_dir, split=args.split, source=args.source, run=args.run,
        kernel_classes=tuple(args.kernel_classes) if args.kernel_classes else None, n_boot=args.bootstrap,
        min_group=args.min_group, device=args.device, overwrite=args.overwrite,
    )
    cols = [c for c in ("group_by", "group", "n_volumes", "macro_auroc", "macro_auroc_ci_low", "macro_auroc_ci_high",
                        "macro_auprc", "macro_f1", "too_small") if c in result.summary.columns]
    print(result.summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
