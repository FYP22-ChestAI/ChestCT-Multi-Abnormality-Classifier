"""One Streamlit widget per script argument, pre-filled with the value the script would really use.

Widget state lives in st.session_state under ``<page key>::<dest>``. Next to it, ``...::default`` remembers
the default the widget was filled with: when a default changes (another experiment picked) and the user
had not touched the field, the field follows the new default; a value the user typed is kept.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from app.argspec import APPEND, FLAG, LIST, VALUE, ArgSpec
from app.defaults import Default

_DOCS = Path(__file__).with_name("arg_docs.yaml")


@lru_cache(maxsize=1)
def _docs() -> dict:
    return yaml.safe_load(_DOCS.read_text(encoding="utf-8")) or {}


def doc(page: str, dest: str) -> dict:
    docs = _docs()
    return {**(docs.get("common") or {}).get(dest, {}), **((docs.get("scripts") or {}).get(page) or {}).get(dest, {})}


def label(arg: ArgSpec, d: dict) -> str:
    text = d.get("label") or arg.flag.lstrip("-").replace("-", " ").capitalize()
    return text + (" *" if arg.required else "")


def _show(value: Any) -> str:
    if value is None or value == [] or value == "":
        return "empty"
    if isinstance(value, (list, tuple)):
        return " ".join(map(str, value))
    return f"{value:g}" if isinstance(value, float) else str(value)


def tooltip(arg: ArgSpec, d: dict, default: Default | None) -> str:
    parts = [d.get("help") or arg.help or ""]
    if d.get("help") and arg.help and arg.help not in d["help"]:
        parts.append(f"*Script help:* {arg.help}")
    if arg.kind != FLAG:
        if default is not None:
            parts.append(f"**Default:** `{_show(default.value)}`  \nfrom `{default.origin}`")
        elif arg.default is not None:
            parts.append(f"**Default:** `{_show(arg.default)}`")
        elif not arg.required:
            parts.append("**Default:** decided by the script when left empty.")
    if arg.required:
        parts.append("**Required.**")
    parts.append(f"Command line: `{arg.flag}`")
    return "\n\n".join(p for p in parts if p)


# ------------------------------------------------------------------ state <-> widget value
def _to_widget(arg: ArgSpec, value: Any, kind: str) -> Any:
    """A default as the widget holds it."""
    if kind == "check":
        return bool(value)
    if kind in ("multi", "lines", "words"):
        items = list(value) if isinstance(value, (list, tuple)) else ([] if value is None else [value])
        return {"multi": items, "lines": "\n".join(map(str, items)), "words": " ".join(map(str, items))}[kind]
    if kind == "text":
        return "" if value is None else str(value)
    if kind == "number" and value is not None:
        return arg.type(value)  # YAML's `100` for a float field: Streamlit refuses mixed int / float
    return value  # select


def _from_widget(arg: ArgSpec, raw: Any, kind: str) -> Any:
    """The widget's state as a value for the command line (None = leave the argument out)."""
    if kind == "check":
        return bool(raw)
    if kind == "multi":
        return list(raw) or None
    if kind in ("lines", "words"):
        items = (raw or "").split("\n" if kind == "lines" else None)
        items = [arg.type(i.strip()) for i in items if i.strip()]
        return items or None
    if kind == "text":
        return arg.type(raw.strip()) if raw and raw.strip() else None
    return raw


def widget_kind(arg: ArgSpec, options: list[str] | None) -> str:
    if arg.kind == FLAG:
        return "check"
    if arg.kind == APPEND:
        return "lines"
    if arg.kind == LIST:
        return "multi" if (arg.choices or options) else "words"
    if arg.choices or options is not None:
        return "select"
    if arg.type in (int, float):
        return "number"
    return "text"


def read_value(page: str, arg: ArgSpec, options: list[str] | None) -> Any:
    key = f"{page}::{arg.dest}"
    if key not in st.session_state:
        return None
    try:
        return _from_widget(arg, st.session_state[key], widget_kind(arg, options))
    except (TypeError, ValueError):
        return None


def sync_default(page: str, arg: ArgSpec, options: list[str] | None, default_value: Any) -> None:
    """Fill a new field with its default; follow a changed default unless the user edited the field."""
    key, dkey = f"{page}::{arg.dest}", f"{page}::{arg.dest}::default"
    kind = widget_kind(arg, options)
    fresh = _to_widget(arg, default_value, kind)
    if key not in st.session_state:
        st.session_state[key], st.session_state[dkey] = fresh, default_value
    elif st.session_state.get(dkey) != default_value:
        if st.session_state[key] == _to_widget(arg, st.session_state.get(dkey), kind):
            st.session_state[key] = fresh
        st.session_state[dkey] = default_value


def edited_values(page: str, args: list[ArgSpec]) -> dict:
    """The fields the user changed on a step's page this session (the pipeline runs with them)."""
    out = {}
    for a in args:
        key, dkey = f"{page}::{a.dest}", f"{page}::{a.dest}::default"
        if key not in st.session_state:
            continue
        # a list in session state is a multiselect; anything else reads the same as a select or a text box
        value = read_value(page, a, [] if isinstance(st.session_state[key], list) else None)
        default = st.session_state.get(dkey)
        if isinstance(value, list) and isinstance(default, (list, tuple)):
            default = list(default)
        if value is not None and value is not False and value != default:
            out[a.dest] = value
    return out


def reset(page: str) -> None:
    for key in [k for k in st.session_state if isinstance(k, str) and k.startswith(f"{page}::")]:
        del st.session_state[key]


def _step(value: Any) -> float:
    if not value:
        return 0.1
    return 10 ** math.floor(math.log10(abs(float(value))))


def render(page: str, arg: ArgSpec, default: Default | None, options: list[str] | None) -> None:
    """Draw the widget; its value is read back with :func:`read_value` (session state)."""
    key = f"{page}::{arg.dest}"
    d = doc(page, arg.dest)
    lbl, hlp = label(arg, d), tooltip(arg, d, default)
    kind = widget_kind(arg, options)
    current = st.session_state.get(key)
    placeholder = "Choose…" if arg.required else "Empty: use the default"

    if kind == "check":
        st.checkbox(lbl, key=key, help=hlp)
    elif kind == "select":
        opts = list(arg.choices or options or [])
        if current not in (None, "") and current not in opts:
            opts.insert(0, current)  # a typed value or a default that is not on disk (yet)
        st.selectbox(lbl, opts, key=key, help=hlp, index=None, placeholder=placeholder,
                     accept_new_options=not arg.choices)
    elif kind == "multi":
        opts = list(arg.choices or options or [])
        opts += [v for v in (current or []) if v not in opts]
        st.multiselect(lbl, opts, key=key, help=hlp, placeholder=placeholder, accept_new_options=not arg.choices)
    elif kind == "number":
        if arg.type is int:
            st.number_input(lbl, key=key, help=hlp, value=None, step=1, placeholder=placeholder)
        else:
            ref = current if current is not None else (default.value if default else arg.default)
            st.number_input(lbl, key=key, help=hlp, value=None, step=_step(ref), format="%g", placeholder=placeholder)
    elif kind == "lines":
        st.text_area(lbl, key=key, help=hlp, placeholder="one per line", height=90)
    elif kind == "words":
        st.text_input(lbl, key=key, help=hlp, placeholder="space-separated, e.g. sharp soft")
    else:
        st.text_input(lbl, key=key, help=hlp, placeholder=placeholder)
