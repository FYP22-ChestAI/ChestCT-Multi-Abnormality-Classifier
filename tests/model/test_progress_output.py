"""Progress lines must reach a pipe (``| tee``, ``nohup``) while a run is still going, not when it ends.

Python block-buffers stdout when it is not a terminal, so a plain print() in a long loop shows nothing
for thousands of volumes (seen on the server with encode_volumes.py). Every test here runs a real
process with stdout connected to a pipe, as on the server."""
import inspect
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from ct_model.utils.log import say

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((REPO / "scripts" / "model").glob("*.py"))


def line_while_running(cmd: list[str], wanted: str, timeout: float = 300.0) -> tuple[str | None, bool]:
    """Start ``cmd`` with stdout on a pipe; return (the first line containing ``wanted``, whether the process
    was still running when that line arrived). The process is killed afterwards."""
    proc = subprocess.Popen(cmd, cwd=REPO, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    found: list[str] = []

    def read():
        for line in proc.stdout:
            if wanted in line:
                found.append(line)
                return

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(timeout)
    running = proc.poll() is None
    proc.kill()
    proc.wait(timeout=60)
    return (found[0] if found else None), running


def test_plain_print_is_held_back_in_a_pipe():
    """Control: proves the harness can see the problem at all."""
    line, running = line_while_running([sys.executable, "-c", "import sys; print('progress'); sys.stdin.read()"],
                                       "progress", timeout=5)
    assert line is None and running


@pytest.mark.parametrize("code", [
    "from ct_model.utils.log import say; import sys; say('progress'); sys.stdin.read()",
    "from ct_model.utils.log import stream_logs; import sys; stream_logs(); print('progress'); sys.stdin.read()",
])
def test_say_and_stream_logs_deliver_lines_while_running(code):
    line, running = line_while_running([sys.executable, "-c", code], "progress", timeout=60)
    assert line == "progress\n" and running


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_every_model_script_makes_stdout_line_buffered(script):
    """Run the script's main() for real (``--help``: it exits right after argument parsing) with stdout on a
    pipe, and report at exit whether stdout had become line-buffered."""
    probe = (
        "import atexit, runpy, sys\n"
        "atexit.register(lambda: sys.stderr.write('LINE_BUFFERED=%s' % sys.stdout.line_buffering))\n"
        f"sys.argv = [{str(script)!r}, '--help']\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert "LINE_BUFFERED=True" in proc.stderr, proc.stderr[-2000:]


def test_train_mil_epochs_appear_while_training(mil_project):
    pytest.importorskip("torch")
    line, running = line_while_running(
        [sys.executable, str(REPO / "scripts/model/train_mil.py"), "--experiment", str(mil_project["exp"]),
         "--max-epochs", "2000", "--patience", "2000"],
        "epoch   1",
    )
    assert line is not None and "val macro AUROC" in line and running


def test_evaluate_mil_groups_appear_while_evaluating(mil_project):
    pytest.importorskip("torch")
    from ct_model.training.config import load_experiment, override
    from ct_model.training.trainer import train_experiment

    run = train_experiment(override(load_experiment(str(mil_project["exp"])), **{"optim.max_epochs": 2}),
                           log=lambda _: None).dir
    line, running = line_while_running(
        [sys.executable, str(REPO / "scripts/model/evaluate_mil.py"), "--experiment-dir", str(run),
         "--bootstrap", "1000000", "--min-group", "10"],  # a bootstrap that takes far longer than the test
        "[1/",
    )
    assert line is not None and "all=all" in line and "bootstrap" in line and running


@pytest.mark.parametrize("path", [
    "ct_model.embeddings.extract:encode_volumes",
    "ct_model.embeddings.extract:encode_slices",
    "ct_model.training.trainer:train_experiment",
    "ct_model.training.evaluate:evaluate_experiment",
])
def test_library_loops_log_with_flush_by_default(path):
    """For callers that are not our scripts (notebooks): the loops' default logger flushes itself."""
    pytest.importorskip("torch")
    module, name = path.split(":")
    fn = getattr(__import__(module, fromlist=[name]), name)
    assert inspect.signature(fn).parameters["log"].default is say
