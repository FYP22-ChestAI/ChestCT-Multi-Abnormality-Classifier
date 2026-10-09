"""The whole pipeline, raw CT -> evaluated, summarized models, as an ordered list of stages.

Each stage runs one registered script. Train and evaluate fan out to one step per experiment x seed, and
encode to one step per encoder the chosen experiments use. Every step gets a status from app/status.py, so
"Run pipeline" only runs what is missing. To add a stage (e.g. LoRA fine-tuning), register its script in
app/registry.py and add one Stage to STAGES, with a status check in app/status.py.

No Streamlit here: the page (app/views/pipeline.py) and the background runner (app/pipeline_runner.py)
both build the plan with :func:`build_steps`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from app import status as S
from app.argspec import display_command
from app.commands import build_args
from app.registry import SCRIPTS, ScriptSpec

AUTO, ALWAYS, SKIP = "auto", "always", "skip"
_SPECS = {s.key: s for s in SCRIPTS}


@dataclass
class Target:
    """What the pipeline aims at, chosen on the Pipeline page."""
    source: str = "ctrate"
    run: str = ""
    experiments: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [0])
    eval_split: str = "test"


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    script_key: str
    check: Callable[[list[str]], S.Status]
    # (step key suffix, step title, the fields the pipeline sets) -- one per step of this stage
    expand: Callable[[Target, Callable], list[tuple[str, str, dict]]]
    shared: bool = True             # one step everything after depends on (preprocessing, encode)
    always_args: dict | None = None  # how "Always run" redoes a done step; None = the stage cannot redo
    long: bool = False              # may take hours: flagged in the confirmation
    group: str = "Model"            # the diagram row it is drawn in


@dataclass
class Step:
    key: str
    stage: str
    title: str
    script_key: str
    args: list[str]
    status: S.Status
    shared: bool
    can_always: bool
    long: bool
    recheck: bool = False           # an earlier shared step runs first: checked again when reached

    @property
    def script(self) -> str:
        return _SPECS[self.script_key].script

    @property
    def command(self) -> str:
        return display_command(self.script, self.args)


def _one(fields: Callable[[Target], dict]):
    return lambda t, make: [("", "", fields(t))]


def _encoders(t: Target, make) -> list[tuple[str, str, dict]]:
    from ct_model.training.config import load_experiment
    names = sorted({load_experiment(e).encoder for e in t.experiments}) or ["dale_ct_2s"]
    return [(n, n, {"source": t.source, "run": t.run, "encoder": n}) for n in names]


def _train_fields(t: Target, exp: str, seed: int) -> dict:
    return {"experiment": exp, "seed": seed, "run": t.run}


def _trainings(t: Target, make) -> list[tuple[str, str, dict]]:
    return [(f"{e}:{s}", f"{e} · seed {s}", _train_fields(t, e, s)) for e in t.experiments for s in t.seeds]


def _evaluations(t: Target, make) -> list[tuple[str, str, dict]]:
    out = []
    for e in t.experiments:
        for s in t.seeds:
            try:
                folder = S.train_run_dir(make("train-mil", _train_fields(t, e, s)))
            except Exception:
                folder = None
            where = folder.as_posix() if folder else f"(the {e} seed {s} run, once trained)"
            out.append((f"{e}:{s}", f"{e} · seed {s}", {"experiment_dir": where, "split": t.eval_split}))
    return out


_RUN = _one(lambda t: {"source": t.source, "run": t.run})
_DATA = "Data preparation"

STAGES: list[Stage] = [
    Stage("worklist", "Plan the run", "make-worklist", S.worklist, _one(lambda t: {"source": t.source, "name": t.run}),
          group=_DATA),
    Stage("ingest", "Ingest", "ingest", S.ingest, _RUN, always_args={}, long=True, group=_DATA),
    Stage("merge", "Merge manifests", "merge-manifests", S.merge, _RUN, always_args={}, group=_DATA),
    Stage("qc", "Quality check", "qc-report", S.qc, _RUN, always_args={}, group=_DATA),
    Stage("splits", "Assign splits", "assign-splits", S.splits, _RUN, group=_DATA),
    Stage("encode", "Encode slices", "encode-volumes", S.encode, _encoders, always_args={}, long=True),
    Stage("train", "Train", "train-mil", S.train, _trainings, shared=False, long=True),
    Stage("evaluate", "Evaluate", "evaluate-mil", S.evaluate, _evaluations, shared=False,
          always_args={"overwrite": True}),
    Stage("summarize", "Summarize", "summarize-experiments", S.always,
          lambda t, make: [("val", "val results", {"split": "val"}), ("test", "test results", {"split": "test"})],
          shared=False),
]


def build_steps(target: Target, page_values: dict[str, dict] | None = None,
                modes: dict[str, str] | None = None) -> list[Step]:
    """Every step for ``target`` with its exact command and current status. ``page_values`` are the fields
    a user changed on each step's own page (by script key); the pipeline's target fields win over them."""
    page_values, modes = page_values or {}, modes or {}

    def make(script_key: str, fields: dict) -> list[str]:
        return build_args(_SPECS[script_key], page_values.get(script_key, {}), fields)

    steps: list[Step] = []
    for stage in STAGES:
        for suffix, title, fields in stage.expand(target, make):
            key = f"{stage.key}:{suffix}" if suffix else stage.key
            if modes.get(key) == ALWAYS and stage.always_args:
                fields = {**fields, **stage.always_args}
            args = make(stage.script_key, fields)
            try:
                status = stage.check(args)
            except Exception as exc:  # a check must never break the page or the runner
                status = S.Status(S.BLOCKED, f"cannot check yet: {exc}")
            steps.append(Step(key, stage.title, title or stage.title, stage.script_key, args, status, stage.shared,
                              stage.always_args is not None, stage.long))
    upstream_runs = False
    for step in steps:
        if upstream_runs and not step.status.needs_run:
            step.recheck = True
        if step.shared and will_run(step, modes):
            upstream_runs = True
    return steps


def will_run(step: Step, modes: dict[str, str]) -> bool:
    mode = modes.get(step.key, AUTO)
    if mode == SKIP or step.status.state == S.NA:
        return False
    if mode == ALWAYS and step.can_always:
        return True
    return step.status.needs_run


def step_dict(step: Step) -> dict:
    d = asdict(step)
    d["status"] = asdict(step.status)
    d["command"] = step.command
    return d
