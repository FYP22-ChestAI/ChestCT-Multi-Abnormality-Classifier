"""Scripts started from the UI run as background jobs: they keep going when the browser tab closes or the
page reruns, and their output goes to a log file anyone can tail.

    outputs/ui_jobs/<job id>.json   what ran (command, pid, start), and exit code + end time once it is over
    outputs/ui_jobs/<job id>.log    everything the script printed

Each job is started through ``python -m app.job_runner`` (in its own session, so a Stop reaches the script
and nothing else), which waits for the script and writes the exit code -- even with no UI open.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.argspec import REPO

JOBS_DIR = REPO / "outputs" / "ui_jobs"


@dataclass
class Job:
    id: str
    script: str
    args: list[str]
    pid: int | None
    started: str
    ended: str | None = None
    exit_code: int | None = None

    @property
    def log(self) -> Path:
        return JOBS_DIR / f"{self.id}.log"

    @property
    def command(self) -> str:
        from app.argspec import display_command
        return display_command(self.script, self.args)

    @property
    def status(self) -> str:
        if self.exit_code is not None:
            return {0: "finished"}.get(self.exit_code, "stopped" if self.exit_code < 0 else "failed")
        if self.pid is None:
            return "starting"
        return "running" if _alive(self.pid) else "lost"  # lost: killed without a trace (e.g. a reboot)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _meta(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def read_job(job_id: str) -> Job | None:
    try:
        return Job(**json.loads(_meta(job_id).read_text(encoding="utf-8")))
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        return None


def write_job(job: Job) -> None:
    tmp = _meta(job.id).with_suffix(".tmp")
    tmp.write_text(json.dumps(job.__dict__, indent=2), encoding="utf-8")
    tmp.replace(_meta(job.id))


def start_job(script: str, args: list[str]) -> Job:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = f"{datetime.now():%Y%m%d-%H%M%S}_{Path(script).stem}"
    while _meta(job_id).exists():  # two starts in one second
        job_id += "x"
    job = Job(id=job_id, script=script, args=list(args), pid=None, started=datetime.now().isoformat(timespec="seconds"))
    write_job(job)
    with open(job.log, "wb") as log:  # the runner records its own pid (= the process group) when it starts
        subprocess.Popen(
            [sys.executable, "-m", "app.job_runner", job_id], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return job


def stop_job(job: Job) -> None:
    """SIGTERM to the job's process group: the runner records the exit and the script stops."""
    if job.pid and job.status == "running":
        os.killpg(job.pid, signal.SIGTERM)


def list_jobs(script: str | None = None) -> list[Job]:
    if not JOBS_DIR.is_dir():
        return []
    jobs = [read_job(p.stem) for p in sorted(JOBS_DIR.glob("*.json"), reverse=True)]
    return [j for j in jobs if j and (script is None or j.script == script)]


def tail(job: Job, max_lines: int = 400) -> str:
    """The last lines of the log; carriage-return progress bars collapse to their latest state."""
    try:
        with open(job.log, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 256_000))
            text = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    lines = [line.rsplit("\r", 1)[-1] for line in text.split("\n")]
    return "\n".join(lines[-max_lines:])


def duration(job: Job) -> str:
    start = datetime.fromisoformat(job.started).timestamp()
    end = datetime.fromisoformat(job.ended).timestamp() if job.ended else time.time()
    s = int(end - start)
    return f"{s // 3600}h {s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m {s % 60:02d}s"
