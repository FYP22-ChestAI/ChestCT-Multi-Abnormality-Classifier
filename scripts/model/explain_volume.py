"""Optional: show the evidence behind one volume's predictions.

    python scripts/model/explain_volume.py --experiment-dir DIR --volume-id valid_1127_a_1
    python scripts/model/explain_volume.py --experiment-dir DIR --volume-id valid_1127_a_1 --labels emphysema --heatmaps

For every chosen label (default: the labels predicted positive at the val thresholds):
  * L1 -- the exact contribution of every slice to the label's logit, and the top-k slices;
  * L2 (--heatmaps) -- a Grad-CAM map inside each of those top slices. This re-encodes them with the
    stage-2 encoder, so use a GPU for DALE (a few slices on CPU take a minute or so).
Writes <run dir>/explain/<volume_id>/: evidence.json and one PNG per label. Needs the HU cache and the
embedding store (i.e. run it on the server). Evidence shows what the model used -- not a validated
localisation (docs/model/README.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ct_preprocessing.cli import friendly_errors, show_settings

from ct_model.utils.log import stream_logs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-dir", required=True)
    ap.add_argument("--volume-id", required=True)
    ap.add_argument("--labels", nargs="+", help="label names (default: the predicted-positive ones)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--heatmaps", action="store_true", help="also compute in-slice Grad-CAM maps (L2)")
    ap.add_argument("--min-cosine", type=float, default=0.99, help="re-encoded vs stored embedding agreement needed for L2")
    ap.add_argument("--run", help="data run holding the volume (default: the experiment's)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return ap


@friendly_errors
def main() -> int:
    stream_logs()  # progress lines reach | tee / log files as they happen
    args = build_parser().parse_args()
    show_settings("explain_volume", vars(args))

    import matplotlib

    matplotlib.use("Agg")
    from ct_model.explain.volume import open_explainer, plot_label
    from ct_model.training.records import ensure_dir, write_json

    ex = open_explainer(args.experiment_dir, device=args.device, run=args.run)
    ev = ex.evidence(args.volume_id)
    names = ev.label_names
    labels = args.labels or [n for n, p in zip(names, ev.predicted) if p]
    unknown = [l for l in labels if l not in names]
    if unknown:
        raise KeyError(f"unknown labels {unknown}; the model's labels are {names}")
    out = ensure_dir(Path(args.experiment_dir) / "explain" / args.volume_id)
    truth = ex.labels(args.volume_id)
    report = {"volume_id": args.volume_id, "run_id": ex.trained.run_info["run_id"],
              "slices": int(ev.contributions.shape[1]), "labels": {}}
    hu = ex.hu(args.volume_id)
    for label in labels:
        c = names.index(label)
        top = ev.top_slices(c, args.top_k)
        maps = ex.heatmaps(args.volume_id, label, top, args.min_cosine) if args.heatmaps else None
        fig = plot_label(ev, label, hu, maps, k=args.top_k, true_value=truth[c])
        fig.savefig(out / f"{label}.png", dpi=110)
        report["labels"][label] = {
            "probability": float(ev.probs[c]), "val_threshold": None if np.isnan(ev.thresholds[c]) else float(ev.thresholds[c]),
            "predicted_positive": bool(ev.predicted[c]), "manifest_label": None if np.isnan(truth[c]) else int(truth[c]),
            "top_slices": [{"slice": int(s), "contribution": float(ev.contributions[c, s])} for s in top],
            "reencode_cosine": [m.reencode_cosine for m in maps] if maps else None,
        }
        print(f"{label}: p={ev.probs[c]:.3f}, top slices {top.tolist()} -> {out / (label + '.png')}")
    if not labels:
        print("no label is predicted positive at the val thresholds -- pass --labels to look at specific ones")
    write_json(out / "evidence.json", report)
    print(f"evidence written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
