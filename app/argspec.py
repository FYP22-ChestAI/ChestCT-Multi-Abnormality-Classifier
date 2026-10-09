"""Read a script's command-line arguments from its own ``build_parser()``, and build the command back.

The UI never keeps its own copy of a script's arguments: add an ``ap.add_argument`` to a script and its
page shows the new field on the next reload. Nothing here imports Streamlit, so it is easy to test.
"""
from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

# how a field is shown and how its value goes back on the command line
FLAG = "flag"        # store_true: a checkbox, "--dry-run" when ticked
LIST = "list"        # nargs="+": "--splits train val"
APPEND = "append"    # action="append": "--add A --add B"
VALUE = "value"      # one value: "--lr 0.0002"


@dataclass(frozen=True)
class ArgSpec:
    flag: str                     # the long option, e.g. "--slice-batch-size"
    dest: str                     # argparse's attribute name, e.g. "slice_batch_size"
    kind: str                     # FLAG | LIST | APPEND | VALUE
    type: type                    # int, float or str (of one element, for LIST / APPEND)
    choices: tuple | None
    required: bool
    default: Any                  # argparse's own default (None for most: the config decides)
    help: str | None
    metavar: str | None


def arg_specs(parser: argparse.ArgumentParser) -> list[ArgSpec]:
    specs = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction) or not action.option_strings:
            continue
        flag = next((s for s in action.option_strings if s.startswith("--")), action.option_strings[0])
        if isinstance(action, argparse._StoreTrueAction):
            kind = FLAG
        elif isinstance(action, argparse._AppendAction):
            kind = APPEND
        elif action.nargs in ("+", "*"):
            kind = LIST
        else:
            kind = VALUE
        specs.append(ArgSpec(
            flag=flag, dest=action.dest, kind=kind, type=action.type or str,
            choices=tuple(action.choices) if action.choices else None, required=bool(action.required),
            default=None if kind == FLAG and action.default is False else action.default,
            help=action.help, metavar=action.metavar if isinstance(action.metavar, str) else None,
        ))
    return specs


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return list(a or []) == list(b or [])
    if isinstance(a, float) or isinstance(b, float):
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return False
    return a == b


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def command_args(specs: list[ArgSpec], values: dict[str, Any], defaults: dict[str, Any]) -> list[str]:
    """The arguments to pass: required fields always, every other field only when it differs from its
    default -- so a bare run uses the script's config fallback, exactly as on the command line."""
    out: list[str] = []
    for s in specs:
        value = values.get(s.dest)
        if value is None or value == "" or (s.kind in (LIST, APPEND) and not value):
            continue
        if s.kind == FLAG:
            if value:
                out.append(s.flag)
            continue
        if not s.required and _same(value, defaults.get(s.dest)):
            continue
        if s.kind == LIST:
            out += [s.flag, *map(_fmt, value)]
        elif s.kind == APPEND:
            for item in value:
                out += [s.flag, _fmt(item)]
        else:
            out += [s.flag, _fmt(value)]
    return out


def display_command(script: str, args: list[str]) -> str:
    return " ".join(["python", script, *map(shlex.quote, args)])


def missing_required(specs: list[ArgSpec], values: dict[str, Any]) -> list[str]:
    return [s.flag for s in specs if s.required and values.get(s.dest) in (None, "", [])]
