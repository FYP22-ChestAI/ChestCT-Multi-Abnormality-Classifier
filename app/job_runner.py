"""Runs one UI job to the end and records its exit code: ``python -m app.job_runner <job id>``.

Started by app.jobs.start_job in a new session with stdout going to the job's log. The script inherits
that log and the process group, so a Stop (SIGTERM to the group) reaches it; the runner itself ignores
the signal, waits for the script, and writes the exit code.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from datetime import datetime

from app.jobs import read_job, write_job


def main(job_id: str) -> int:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)  # the script gets the group's SIGTERM; we record it
    job = read_job(job_id)
    job.pid = os.getpid()  # we lead the job's process group (start_new_session)
    write_job(job)
    print(f"$ {job.command}\n", flush=True)
    child = subprocess.Popen([sys.executable, job.script, *job.args],
                             preexec_fn=lambda: signal.signal(signal.SIGTERM, signal.SIG_DFL))
    code = child.wait()
    job.exit_code, job.ended = code, datetime.now().isoformat(timespec="seconds")
    write_job(job)
    print(f"\n[{'finished' if code == 0 else 'stopped' if code < 0 else 'failed'}: exit code {code}]", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
