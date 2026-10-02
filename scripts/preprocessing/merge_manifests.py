"""Step 3: merge the per-chunk manifests into data/manifest.csv.

    python scripts/preprocessing/merge_manifests.py

One row per volume across every source that has been ingested, deduplicated by
volume_id. Splits already frozen by assign_splits.py are joined back in; patients
without one yet are "unassigned". Also writes data/preprocessing_manifest.json
(the settings and fingerprint that produced the cache, and what failed).

For an archive source, give the checklist of every patient folder on the
original disk (one name per line, e.g. made with `dir /b /ad`) to see which
patients never arrived:

    python scripts/preprocessing/merge_manifests.py --patients-file patients_on_disk.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ct_preprocessing.cli import add_config_arg, friendly_errors, show_settings
from ct_preprocessing.config import load_config
from ct_preprocessing.ingest.merge import (
    merge_sources, read_patients_file, reconcile_patients, suspicious_patients, write_run_record,
)
from ct_preprocessing.manifest import FOLDER_BUILDER


@friendly_errors
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--source", action="append", default=None, help="a source to merge (repeatable; default: every configured source with chunk manifests)")
    ap.add_argument("--patients-file", default=None, help="checklist of patient folders on the source disk, to report missing ones")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sources = args.source or list(cfg.sources)
    show_settings("using", dict(sources=",".join(sources), manifest=cfg.paths.manifest_path))

    manifest, report = merge_sources(cfg, sources)
    out = Path(cfg.paths.manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    write_run_record(cfg.paths.preprocessing_manifest_path, cfg, manifest, report)

    print(f"wrote {len(manifest)} rows to {out}: " + ", ".join(f"{k}={v}" for k, v in report.rows_per_source.items()))
    if report.duplicates_dropped:
        print(f"dropped {report.duplicates_dropped} duplicate volume(s) (the same scan arrived more than once)")
    if report.failures:
        print(f"{len(report.failures)} volume(s) failed processing -- see {cfg.paths.preprocessing_manifest_path}")
    print("splits: " + ", ".join(f"{k}={v}" for k, v in manifest["split"].value_counts().items()))
    for name in report.rows_per_source:
        if cfg.sources[name].manifest_builder == FOLDER_BUILDER:
            for patient, n in suspicious_patients(manifest[manifest["source_name"] == name])[:3]:
                print(
                    f"WARNING [{name}]: patient {patient!r} has {n} scans, far more than the rest. "
                    "Was an archive zipped one folder too high? The patient folders must sit directly inside the archive."
                )

    if args.patients_file:
        folder_rows = manifest[manifest["source_name"].map(lambda n: cfg.sources[n].manifest_builder == FOLDER_BUILDER)]
        missing, unexpected = reconcile_patients(folder_rows, read_patients_file(args.patients_file))
        print(f"checklist: {len(missing)} patient folder(s) missing from the manifest, {len(unexpected)} not on the checklist")
        for name in missing[:20]:
            print(f"  MISSING {name}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return 1 if missing else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
