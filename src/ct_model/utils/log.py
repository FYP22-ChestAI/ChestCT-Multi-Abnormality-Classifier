"""Progress output that shows up while a long run is going, not at its end.

When stdout is a pipe or a file (``| tee -a run.log``, ``nohup``, ``> run.log``), Python buffers it in
8 KB blocks, so short progress lines would appear only in bursts of ~80 lines -- or when the process
exits. Two layers, so nothing depends on remembering ``flush=True``:

* every script under scripts/model calls ``stream_logs()`` first: stdout then flushes at every newline,
  whoever prints (our code, numpy / torch / timm warnings printed to stdout, ...);
* library loops log through ``say``, which flushes itself -- for callers that are not our scripts
  (notebooks, other code).
"""
from __future__ import annotations

import sys


def stream_logs() -> None:
    """Make stdout line-buffered for the rest of the process (call first in a script's main())."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:  # absent when stdout was replaced by a non-TextIOWrapper object
        reconfigure(line_buffering=True)


def say(message: str) -> None:
    print(message, flush=True)
