"""Export finished runs, seed ensembles and LoRA adaptations from outputs/ to an MLflow server (e.g. DagsHub).

    python scripts/model/export_mlflow.py --dry-run          # what would be exported
    python scripts/model/export_mlflow.py                    # export what the server does not have yet
    python scripts/model/export_mlflow.py --refresh --name abmil_perlabel_mixed_dale2s   # re-export after new evals

The server and login come from the standard MLflow environment variables (never put them in the repo):

    export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
    export MLFLOW_TRACKING_USERNAME=<dagshub user>
    export MLFLOW_TRACKING_PASSWORD=<dagshub access token>

The run folders stay the source of truth: the export is idempotent (an exported run is recognised by its
``ct.run_id`` tag and skipped), so run it after every phase. Exported: configs, curves, val / test / clinical
metrics and reports, and the trained weights (best.pt; LoRA adapter.safetensors + head.pt). Never exported: the HU
cache, embedding stores, resume checkpoints, and -- unless --with-predictions -- per-volume prediction files
(CT-RATE patient-level data and labels, CC BY-NC-SA). See ct_model.tracking.mlflow_export.
"""
from __future__ import annotations

import argparse
import os

from ct_preprocessing.cli import friendly_errors, show_settings

from ct_model.utils.log import stream_logs


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracking-uri", help="default: $MLFLOW_TRACKING_URI")
    ap.add_argument("--experiments-dir", default="outputs/experiments")
    ap.add_argument("--adapters-dir", default="outputs/adapters")
    ap.add_argument("--name", help="only this experiment / adapt config name")
    ap.add_argument("--refresh", action="store_true", help="delete and re-export runs the server already has")
    ap.add_argument("--no-weights", action="store_true", help="skip best.pt / adapter weights")
    ap.add_argument("--with-predictions", action="store_true",
                    help="also per-volume prediction files (CT-RATE patient-level data: only to a PRIVATE server)")
    ap.add_argument("--dry-run", action="store_true", help="list what would be exported; nothing is sent")
    return ap


@friendly_errors
def main() -> int:
    stream_logs()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # a local file: URI (tests, --dry-run) is fine here
    args = build_parser().parse_args()
    uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri and not args.dry_run:
        raise SystemExit("no MLflow server: set MLFLOW_TRACKING_URI (and MLFLOW_TRACKING_USERNAME / _PASSWORD) or pass --tracking-uri")
    show_settings("export_mlflow", {**{k: v for k, v in vars(args).items() if k != "tracking_uri"}, "tracking_uri": uri,
                                    "login": "set" if os.environ.get("MLFLOW_TRACKING_PASSWORD") else "not set"})
    from ct_model.tracking.mlflow_export import export_outputs

    summary = export_outputs(args.experiments_dir, args.adapters_dir, name=args.name,
                             tracking_uri=uri or "file:./outputs/.mlflow_dry_run", refresh=args.refresh,
                             with_weights=not args.no_weights, with_predictions=args.with_predictions, dry_run=args.dry_run)
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
