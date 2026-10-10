"""Field values -> config defaults, dropdown options and the command to run, for any registered script.

Shared by the script pages (values from the widgets) and the pipeline (the values a user changed on a
step's page, plus the pipeline's own target fields). No Streamlit here.
"""
from __future__ import annotations

import importlib.util
from functools import lru_cache
from types import ModuleType

from app.argspec import REPO, ArgSpec, arg_specs, command_args
from app.defaults import Default
from app.registry import ScriptSpec


@lru_cache(maxsize=64)
def _module(script: str, mtime: float) -> ModuleType:
    path = REPO / script
    spec = importlib.util.spec_from_file_location(f"ui_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # scripts only run main() under __main__, so importing is safe
    return module


def script_module(script: str) -> ModuleType:
    """The script imported as a module (reloaded when the file changes)."""
    return _module(script, (REPO / script).stat().st_mtime)


def script_info(script: str) -> tuple[str, list[ArgSpec]]:
    """The script's description (its docstring) and its arguments, from its own build_parser()."""
    module = script_module(script)
    if not hasattr(module, "build_parser"):
        raise AttributeError(f"{script} has no build_parser() -- move its argparse setup into one")
    parser = module.build_parser()
    return parser.description or "", arg_specs(parser)


def parse(script: str, argv: list[str]):
    """The argparse namespace the script would see for ``argv``."""
    return script_module(script).build_parser().parse_args(argv)


def resolve(spec: ScriptSpec, args: list[ArgSpec], values: dict,
            on_error=None) -> tuple[dict[str, Default], dict[str, list[str]]]:
    """Config defaults and dropdown options for these field values (None = not set). Resolved twice, so
    a default that depends on another default (the run of the encode config's source) settles."""
    defaults: dict[str, Default] = {}
    options: dict[str, list[str]] = {}
    for _ in range(2):
        ctx = {a.dest: values.get(a.dest) if values.get(a.dest) not in (None, [], "") else
               (defaults[a.dest].value if a.dest in defaults else a.default) for a in args}
        if spec.defaults:
            try:
                defaults = spec.defaults(ctx)
            except Exception as exc:  # a broken config must not take the page down
                if on_error:
                    on_error(f"Could not read the config defaults: {exc}")
                defaults = {}
        for dest, provider in spec.choices.items():
            try:
                options[dest] = provider(ctx)
            except Exception:
                options[dest] = []
        for a in args:  # a required pick (experiment, source): start at the first option, so the rest fills in
            pool = list(a.choices or options.get(a.dest) or [])
            if a.required and a.dest not in defaults and a.default is None and pool:
                defaults[a.dest] = Default(pool[0], "the first available option")
    return defaults, options


def flat(args: list[ArgSpec], defaults: dict[str, Default]) -> dict:
    return {a.dest: (defaults[a.dest].value if a.dest in defaults else a.default) for a in args}


def build_args(spec: ScriptSpec, values: dict, forced: dict | None = None) -> list[str]:
    """The arguments for ``values`` (only those that differ from the defaults), plus every ``forced``
    field written out explicitly -- the pipeline's run / experiment / seed, so its commands are unambiguous."""
    _, args = script_info(spec.script)
    known = {a.dest for a in args}
    merged = {**values, **{k: v for k, v in (forced or {}).items() if k in known}}
    defaults, _ = resolve(spec, args, merged)
    base = flat(args, defaults)
    for dest in forced or {}:
        base.pop(dest, None)  # never "equal to the default": always on the command line
    return command_args(args, merged, base)
