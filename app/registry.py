"""Every script the UI can run, grouped by phase: THE place to edit when the project grows.

To add a script: give it a ``build_parser()`` (like scripts/model/train_mil.py) and add one ScriptSpec
below. Its fields, help texts and argparse defaults are read from the script itself; ``defaults`` only
says where the config fallbacks live, ``choices`` which fields get a dropdown from disk. To add a phase,
add it to PHASES and use it in a ScriptSpec. Plain-language labels and tooltips go in arg_docs.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app import choices as c
from app import defaults as d

PHASES = [
    "1 · Preprocessing",
    "2 · Encode slices (stage 2)",
    "3 · Train & evaluate (stages 3–4)",
]


@dataclass(frozen=True)
class ScriptSpec:
    key: str                      # page URL and widget-key prefix; unique
    phase: str                    # one of PHASES
    title: str                    # page title
    script: str                   # path from the repo root
    summary: str                  # one line, shown on the home page and under the title
    icon: str = ":material/terminal:"
    optional: bool = False        # a side tool, not a step of the main flow
    essential: tuple[str, ...] = ()   # fields shown at the top (required ones always are); others under "Advanced"
    defaults: d.Resolver | None = None
    choices: dict[str, Callable[[dict], list[str]]] = field(default_factory=dict)


_PRE, _ENC, _TRAIN = PHASES

SCRIPTS: list[ScriptSpec] = [
    # ---------------------------------------------------------------- 1 · preprocessing
    ScriptSpec("make-worklist", _PRE, "Plan a run (worklist)", "scripts/preprocessing/make_worklist.py",
               "CT-RATE only: choose exactly which volumes a new run will ingest.",
               icon=":material/checklist:", essential=("source", "name", "train_kernel"),
               defaults=d.worklist_defaults, choices={"source": c.sources, "from_run": c.runs}),
    ScriptSpec("ingest", _PRE, "Ingest", "scripts/preprocessing/ingest.py",
               "Fetch → preprocess → delete raw data, chunk by chunk. Long-running and resumable.",
               icon=":material/download:", essential=("source", "run", "max_chunks"),
               defaults=d.ingest_defaults, choices={"source": c.sources, "run": c.runs}),
    ScriptSpec("merge-manifests", _PRE, "Merge manifests", "scripts/preprocessing/merge_manifests.py",
               "Join the per-chunk manifests (plus labels and splits) into the run's manifest.csv.",
               icon=":material/merge:", essential=("source", "run"),
               choices={"source": c.sources, "run": c.runs}),
    ScriptSpec("qc-report", _PRE, "Quality check", "scripts/preprocessing/qc_report.py",
               "Re-check every cached volume and write qc_report.csv and montage images.",
               icon=":material/fact_check:", essential=("source", "run"),
               choices={"source": c.sources, "run": c.runs}),
    ScriptSpec("assign-splits", _PRE, "Assign splits", "scripts/preprocessing/assign_splits.py",
               "Freeze train / val / test by patient (once). Only QC-passed volumes count.",
               icon=":material/call_split:", essential=("source", "run", "dry_run"),
               defaults=d.splits_defaults, choices={"source": c.sources, "run": c.runs}),
    ScriptSpec("make-kernel-table", _PRE, "Kernel table (draft)", "scripts/preprocessing/make_kernel_table.py",
               "Draft the sharp / soft kernel table from CT-RATE metadata.",
               icon=":material/table:", optional=True, defaults=d.kernel_table_defaults),
    ScriptSpec("kernel-survey", _PRE, "Kernel survey", "scripts/preprocessing/kernel_survey.py",
               "Check the sharp / soft kernel table by measurement on a survey run.",
               icon=":material/query_stats:", optional=True, essential=("source", "run"),
               choices={"source": c.sources, "run": c.runs}),
    # ---------------------------------------------------------------- 2 · encode
    ScriptSpec("check-encoder", _ENC, "Check encoder", "scripts/model/check_encoder.py",
               "Smoke-test a backbone on one volume (speed, memory, output sanity). Writes nothing.",
               icon=":material/health_and_safety:", essential=("encoder", "run", "device"),
               defaults=d.stage2_defaults,
               choices={"encoder": c.encoders, "source": c.sources, "run": c.runs}),
    ScriptSpec("encode-volumes", _ENC, "Encode volumes", "scripts/model/encode_volumes.py",
               "Write per-slice embeddings for a run into the embedding store. Resumable.",
               icon=":material/memory:", essential=("encoder", "run", "dry_run"),
               defaults=d.stage2_defaults,
               choices={"encoder": c.encoders, "source": c.sources, "run": c.runs,
                        "kernel_classes": c.kernel_classes}),
    # ---------------------------------------------------------------- 3 · train & evaluate
    ScriptSpec("train-mil", _TRAIN, "Train", "scripts/model/train_mil.py",
               "Train an aggregator + head on the frozen embeddings (train + val only).",
               icon=":material/model_training:", essential=("experiment", "seed"),
               defaults=d.train_defaults, choices={"experiment": c.experiments, "run": c.runs}),
    ScriptSpec("evaluate-mil", _TRAIN, "Evaluate", "scripts/model/evaluate_mil.py",
               "Evaluate a finished run (normally on test, once) with bootstrap CIs.",
               icon=":material/rule:", essential=("experiment_dir", "split"),
               defaults=d.trained_run_defaults,
               choices={"experiment_dir": c.experiment_dirs, "run": c.runs, "kernel_classes": c.kernel_classes}),
    ScriptSpec("summarize-experiments", _TRAIN, "Summarize experiments",
               "scripts/model/summarize_experiments.py",
               "Group results by configuration and average over seeds (writes summary_<split>.csv).",
               icon=":material/summarize:", essential=("split", "name")),
    ScriptSpec("explain-volume", _TRAIN, "Explain a volume", "scripts/model/explain_volume.py",
               "Optional evidence for one volume: slice contributions and Grad-CAM heatmaps.",
               icon=":material/visibility:", optional=True, essential=("experiment_dir", "volume_id", "labels"),
               defaults=d.trained_run_defaults,
               choices={"experiment_dir": c.experiment_dirs, "volume_id": c.volume_ids, "labels": c.labels,
                        "run": c.runs}),
]


def by_phase() -> dict[str, list[ScriptSpec]]:
    return {phase: [s for s in SCRIPTS if s.phase == phase] for phase in PHASES}
