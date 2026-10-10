"""Runs the whole pipeline as one background job: ``python app/pipeline_runner.py <plan.json>``.

Started from the Pipeline page through app/jobs.py, so it has a log, a Stop button and a Jobs entry. The
plan holds the target, the per-page settings and the per-step modes. Before every step the plan is built
again from disk, so a step that an earlier step (or another job) already completed is skipped, and the
commands use the outputs as they are now (e.g. the training run folder after an upstream change).

Progress goes to <plan>.progress.json for the page. The first failure stops the pipeline.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.pipeline import Target, build_steps, step_dict, will_run  # noqa: E402
from app.status import BLOCKED  # noqa: E402


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def progress_path(plan_path: str | Path) -> Path:
    return Path(str(plan_path).removesuffix(".plan.json") + ".progress.json")


def read_progress(plan_path: str | Path) -> dict:
    try:
        return json.loads(progress_path(plan_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class Progress:
    def __init__(self, plan_path: Path):
        self.path = progress_path(plan_path)
        self.data = {"started": now(), "ended": None, "result": None, "current": None, "steps": {}, "order": []}

    def plan(self, steps, modes) -> None:
        """Refresh the steps not started yet: what they will do, given the outputs on disk now."""
        self.data["order"] = [s.key for s in steps]
        for s in steps:
            entry = self.data["steps"].get(s.key)
            if entry and entry["state"] not in ("pending", "skipped"):
                continue
            self.data["steps"][s.key] = {**step_dict(s), "state": "pending" if will_run(s, modes) else "skipped",
                                         "started": None, "ended": None, "exit_code": None}
        self.write()

    def mark(self, key: str, **fields) -> None:
        self.data["steps"][key].update(fields)
        self.write()

    def write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.path)


def main(plan_path: str) -> int:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    target, pages, modes = Target(**plan["target"]), plan.get("page_values", {}), plan.get("modes", {})
    progress = Progress(Path(plan_path))

    def stopped(signum, frame):  # Stop on the page: the current step got the same signal
        if progress.data["current"]:
            progress.mark(progress.data["current"], state="stopped", ended=now())
        progress.data.update(ended=now(), result="stopped")
        progress.write()
        print("\n[pipeline stopped]", flush=True)
        os._exit(143)

    signal.signal(signal.SIGTERM, stopped)
    attempted: set[str] = set()
    n = 0
    while True:
        steps = build_steps(target, pages, modes)
        progress.plan(steps, modes)
        todo = [s for s in steps if s.key not in attempted and will_run(s, modes)]
        if not todo:
            break
        step = todo[0]
        attempted.add(step.key)
        n += 1
        if step.status.state == BLOCKED:
            print(f"\n=== {step.stage}: {step.title} cannot run: {step.status.detail}", flush=True)
            progress.mark(step.key, state="failed", detail=step.status.detail, ended=now())
            progress.data.update(ended=now(), result="failed")
            progress.write()
            return 1
        print(f"\n{'=' * 100}\n=== step {n} ({len(todo) - 1} more planned): {step.stage} · {step.title}\n"
              f"=== {step.status.detail}\n$ {step.command}\n{'=' * 100}", flush=True)
        progress.data["current"] = step.key
        progress.mark(step.key, state="running", started=now(), command=step.command)
        code = subprocess.call([sys.executable, step.script, *step.args], cwd=REPO)
        progress.mark(step.key, state="done" if code == 0 else "failed", ended=now(), exit_code=code)
        progress.data["current"] = None
        if code != 0:
            print(f"\n[pipeline stopped: {step.stage} · {step.title} failed with exit code {code}. "
                  "Fix it and press Run again: finished steps are skipped.]", flush=True)
            progress.data.update(ended=now(), result="failed")
            progress.write()
            return code
    progress.data.update(ended=now(), result="finished")
    progress.write()
    print(f"\n[pipeline finished: {n} step(s) run, everything else was already done]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
