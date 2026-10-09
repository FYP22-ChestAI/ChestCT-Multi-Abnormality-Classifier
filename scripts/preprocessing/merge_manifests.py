"""Step 3: merge a run's per-chunk manifests into the run's manifest.csv.

    python scripts/preprocessing/merge_manifests.py --source ctrate --run train-sharp

One row per volume of the run (data/runs/<source>/<run>/manifest.csv), deduplicated by volume_id,
with the volume's kernel and kernel class (sharp / soft / other), optional label_* columns, and the
patient-level split. Splits already frozen by assign_splits.py are joined back in; patients without
one yet are "unassigned". Also writes preprocessing_manifest.json next to it (the settings and
fingerprint that produced the cache, and what failed).

Labels (never used by preprocessing or splitting, only joined here):
  * ctrate     -- the label CSVs make_worklist.py downloaded into data/metadata/ (predicted labels);
  * nhrd_local -- the labels CSV named in sources.nhrd_local.labels.file, read from the same Drive
                  folder as the archives, or a local copy given with --labels-file. A source with no
                  labels, or labels that cannot be read, is merged without them (a warning says so).

For an archive source, give the checklist of every patient folder on the original disk (one name per
line, e.g. made with `dir /b /ad`) to see which patients never arrived:

    python scripts/preprocessing/merge_manifests.py --source nhrd_local --patients-file patients_on_disk.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ct_preprocessing.cli import add_config_arg, friendly_errors, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.ingest.archives import make_backend
from ct_preprocessing.ingest.labels import fetch_archive_labels, load_ctrate_labels, read_labels_csv
from ct_preprocessing.ingest.merge import merge_run, read_patients_file, reconcile_patients, suspicious_patients, write_run_record
from ct_preprocessing.ingest.sources import archive_remote
from ct_preprocessing.manifest import CTRATE_BUILDER, FOLDER_BUILDER
from ct_preprocessing.runs import DEFAULT_ARCHIVE_RUN, resolve_run


def _labels_for(cfg, source_name, source_cfg, labels_file):
    """(labels frame or None, manifest key, labels key). Optional: any problem becomes a warning."""
    if source_cfg.manifest_builder == CTRATE_BUILDER:
        frame = load_ctrate_labels(cfg.paths)
        if frame is None:
            print("warning: no CT-RATE label files in data/metadata -- merging without labels (make_worklist.py downloads them)")
        return frame, "volume_id", "volume_id"
    try:
        if labels_file:
            path = Path(labels_file)
        elif source_cfg.labels.file:
            path = fetch_archive_labels(
                make_backend(archive_remote(cfg, source_name)), source_cfg.labels,
                Path(cfg.paths.metadata_dir) / f"{source_name}_labels.csv",
            )
        else:
            return None, "", ""
        frame = read_labels_csv(path)
    except Exception as exc:  # noqa: BLE001 - labels are optional
        print(f"warning: labels not read ({type(exc).__name__}: {str(exc)[:200]}) -- merging without labels")
        return None, "", ""
    key = source_cfg.labels.key
    return frame.rename(columns={frame.columns[0]: key}), key, key


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--source", required=True, help="the source of the run to merge (ctrate, nhrd_local, ...)")
    ap.add_argument("--run", default=None, help="the run to merge (default: the only one)")
    ap.add_argument("--labels-file", default=None, help="a local labels CSV, instead of reading it from the source")
    ap.add_argument("--patients-file", default=None, help="checklist of patient folders on the source disk, to report missing ones")
    return ap


@friendly_errors
def main() -> int:
    args = build_parser().parse_args()

    cfg = load_config(args.config)
    source_cfg = cfg.source(args.source)
    run = resolve_run(cfg.paths, args.source, args.run, create_default=None)
    show_settings("using", dict(source=args.source, run=run.name, manifest=run.manifest_path))

    labels, manifest_key, labels_key = _labels_for(cfg, args.source, source_cfg, args.labels_file)
    manifest, report = merge_run(cfg, run, labels=labels, manifest_key=manifest_key or "volume_id", labels_key=labels_key or "volume_id")
    run.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(run.manifest_path, index=False)
    write_run_record(run, cfg, manifest, report)

    print(f"wrote {report.rows} rows to {run.manifest_path}")
    print("kernel classes: " + ", ".join(f"{k}={v}" for k, v in sorted(report.kernel_classes.items())))
    if report.labels:
        print(report.labels.format())
    if report.duplicates_dropped:
        print(f"dropped {report.duplicates_dropped} duplicate volume(s) (the same scan arrived more than once)")
    if report.failures:
        print(f"{len(report.failures)} volume(s) failed processing -- see {run.record_path}")
    print("splits: " + ", ".join(f"{k}={v}" for k, v in manifest["split"].value_counts().items()))
    if source_cfg.manifest_builder == FOLDER_BUILDER:
        for patient, n in suspicious_patients(manifest)[:3]:
            print(
                f"WARNING: patient {patient!r} has {n} scans, far more than the rest. "
                "Was an archive zipped one folder too high? The patient folders must sit directly inside the archive."
            )

    if args.patients_file:
        missing, unexpected = reconcile_patients(manifest, read_patients_file(args.patients_file))
        print(f"checklist: {len(missing)} patient folder(s) missing from the manifest, {len(unexpected)} not on the checklist")
        for name in missing[:20]:
            print(f"  MISSING {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return 1 if missing else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
