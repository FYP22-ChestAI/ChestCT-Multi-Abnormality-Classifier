"""Check the sharp / soft kernel table by measurement.

    python scripts/preprocessing/make_worklist.py --survey 5          # a small run: 5 scan pairs per kernel pair
    python scripts/preprocessing/ingest.py --source ctrate --run kernel-survey
    python scripts/preprocessing/kernel_survey.py --source ctrate --run kernel-survey

Each surveyed scan has two reconstructions of the same raw data. For every pair this measures how much
fine detail and noise each volume has (ct_preprocessing.kernels.sharpness_score, a mean in-plane
Laplacian on lung and soft-tissue voxels) and records which one is the sharper. Per (manufacturer,
kernel) pair it then reports how often that kernel was the sharper one, next to the class currently
in configs/kernel_classes.csv. A kernel that is "sharp" in the table should be the sharper one in
(nearly) every pair, a "soft" one in (nearly) none; rows that disagree are flagged for a human to look at.
The table is never changed by this script.

Writes kernel_survey.csv (one row per pair) and kernel_survey_summary.csv into the run's folder.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ct_preprocessing.cli import add_config_arg, friendly_errors, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.ingest.worklist import read_worklist
from ct_preprocessing.kernels import classify, load_kernel_table, sharpness_score
from ct_preprocessing.runs import resolve_run

AGREE = 0.8  # share of pairs a kernel must win (or lose) to count as clearly sharp (or soft)


@friendly_errors
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--source", default="ctrate", help="a CT-RATE-type source (default: ctrate)")
    ap.add_argument("--run", default="kernel-survey", help="the survey run (default: kernel-survey)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = resolve_run(cfg.paths, args.source, args.run)
    show_settings("using", dict(source=args.source, run=run.name, cache=cfg.paths.cache_dir))
    table = load_kernel_table(cfg.paths.kernel_table)
    worklist = read_worklist(run.worklist_path)

    pairs = []
    for (patient, scan), group in worklist.groupby(["patient_id", "scan_id"]):
        if len(group) != 2:
            continue
        scores, ok = [], True
        for volume_id in group["volume_id"]:
            path = f"{cfg.paths.cache_dir}/{volume_id}.npy"
            try:
                scores.append(sharpness_score(np.load(path, mmap_mode="r")))
            except OSError:
                ok = False
        if not ok:
            continue
        a, b = group.iloc[0], group.iloc[1]
        sharper = a if scores[0] > scores[1] else b
        pairs.append({
            "patient_id": patient, "scan_id": scan, "manufacturer": a["manufacturer"],
            "kernel_a": a["kernel"], "kernel_b": b["kernel"], "score_a": scores[0], "score_b": scores[1],
            "sharper_kernel": sharper["kernel"],
        })
    if not pairs:
        raise ValueError(f"no surveyed scan has both reconstructions cached -- run ingest.py --source {args.source} --run {run.name} first")

    detail = pd.DataFrame(pairs)
    detail.to_csv(run.dir / "kernel_survey.csv", index=False)

    rows = []
    for _, p in detail.iterrows():
        for kernel, other in ((p["kernel_a"], p["kernel_b"]), (p["kernel_b"], p["kernel_a"])):
            if kernel == other:
                continue  # two reconstructions with the same kernel say nothing about it
            rows.append({"manufacturer": p["manufacturer"], "kernel": kernel, "wins": int(p["sharper_kernel"] == kernel)})
    summary = (
        pd.DataFrame(rows).groupby(["manufacturer", "kernel"])["wins"].agg(pairs="size", sharper="sum").reset_index()
    )
    summary["share_sharper"] = (summary["sharper"] / summary["pairs"]).round(2)
    summary["table_class"] = [classify(m, k, table) for m, k in zip(summary["manufacturer"], summary["kernel"])]

    def verdict(row) -> str:
        measured = "sharp" if row["share_sharper"] >= AGREE else "soft" if row["share_sharper"] <= 1 - AGREE else "unclear"
        if row["table_class"] == "other" or measured == "unclear":
            return f"check ({measured})"
        return "ok" if measured == row["table_class"] else f"DISAGREES (measured {measured})"

    summary["verdict"] = summary.apply(verdict, axis=1)
    summary = summary.sort_values("pairs", ascending=False)
    summary.to_csv(run.dir / "kernel_survey_summary.csv", index=False)

    print(f"{len(detail)} scan pairs measured")
    print(summary.to_string(index=False))
    bad = summary[~summary["verdict"].eq("ok")]
    print(f"\n{len(summary) - len(bad)} of {len(summary)} kernels agree with configs/kernel_classes.csv; {len(bad)} need a look")
    print(f"wrote {run.dir / 'kernel_survey.csv'} and {run.dir / 'kernel_survey_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
