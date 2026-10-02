"""Small helpers shared by the command-line scripts in scripts/preprocessing/.

Every script follows the same rule: a value comes from the config file unless
the command line overrides it, and the settings actually used are printed as
the very first line, so a bare run is never ambiguous.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
from typing import Any, Callable

from .config import DEFAULT_CONFIG_PATH


def add_config_arg(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH, help=f"settings file (default: {DEFAULT_CONFIG_PATH})")


def pick(cli_value: Any, config_value: Any) -> Any:
    """The command-line value if one was given, else the config default."""
    return config_value if cli_value is None else cli_value


def show_settings(title: str, settings: dict) -> None:
    """Print the effective settings as one line, flagging nothing -- just the facts."""
    shown = ", ".join(f"{k}={v}" for k, v in settings.items())
    print(f"{title}: {shown}", flush=True)


def friendly_errors(main: Callable[[], int]) -> Callable[[], int]:
    """Turn the errors a user can cause (a missing file or worklist, a bad config
    value, an unreachable source, a refused scratch folder) into one clean
    ``error: ...`` line and exit code 1, instead of a traceback. Set CT_DEBUG=1
    to get the traceback back when chasing a real bug."""

    @functools.wraps(main)
    def wrapper(*args, **kwargs) -> int:
        try:
            return main(*args, **kwargs)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
            if os.environ.get("CT_DEBUG"):
                raise
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            print(f"error: {message}", file=sys.stderr)
            return 1

    return wrapper
