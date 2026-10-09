"""The one page every script uses: what the step does, its fields, the exact command, Run, and the live log."""
from __future__ import annotations

import streamlit as st

from app import form, nav
from app.argspec import FLAG, command_args, display_command, missing_required
from app.commands import flat, resolve, script_info
from app.jobs import list_jobs, start_job
from app.registry import ScriptSpec
from app.views.jobs import job_panel


def _summary_and_details(doc: str) -> tuple[str, str]:
    """The docstring's first line, and the rest (usage examples, explanation)."""
    first, _, rest = doc.strip().partition("\n")
    return first.strip(), rest.strip("\n")


def _values(spec: ScriptSpec, args, options) -> dict:
    return {a.dest: form.read_value(spec.key, a, options.get(a.dest)) for a in args}


# flags that overwrite or freeze something: the Run button asks first
_RISKY = {
    "force": "overwrites the existing file, which may hold reviewed changes",
    "overwrite": "replaces an evaluation that already exists",
    "allow_new_settings": "lets changed preprocessing settings replace the ones the cache was built with",
    "retry_failed": "re-runs chunks that finished with failed volumes",
}


def _risks(spec: ScriptSpec, values: dict) -> list[str]:
    risks = [f"`--{dest.replace('_', '-')}` {why}" for dest, why in _RISKY.items() if values.get(dest)]
    if spec.key == "assign-splits" and not values.get("dry_run"):
        risks.append("splits are **frozen once written**: tick *Dry run* first to check the counts")
    return risks


def _start(spec: ScriptSpec, cmd: list[str]) -> None:
    job = start_job(spec.script, cmd)
    st.session_state[f"{spec.key}::job"] = job.id
    st.toast(f"Started {job.id}. It keeps running if you close this tab.")


@st.dialog("Please confirm", icon=":material/warning:")
def _confirm_risky(spec: ScriptSpec, cmd: list[str], risks: list[str]) -> None:
    st.markdown("This run:\n" + "\n".join(f"- {r}" for r in risks))
    st.code(display_command(spec.script, cmd), language="bash", wrap_lines=True)
    if st.button("Run anyway", type="primary", width="stretch"):
        _start(spec, cmd)
        st.rerun()


def script_page(spec: ScriptSpec) -> None:
    doc, args = script_info(spec.script)
    first, details = _summary_and_details(doc)

    st.title(spec.title)
    st.caption(f"{spec.phase}  ·  `{spec.script}`")
    st.markdown(spec.summary)
    with st.expander("What this step does (the script's full description)"):
        st.markdown(f"**{first}**")
        st.code(details or "(no further description)", language=None, wrap_lines=True)

    defaults, options = resolve(spec, args, _values(spec, args, {}), on_error=st.warning)
    for a in args:  # fill and update the fields before drawing them
        d = defaults.get(a.dest)
        form.sync_default(spec.key, a, options.get(a.dest), d.value if d else a.default)

    essential = [a for a in args if a.required or a.dest in spec.essential]
    essential.sort(key=lambda a: (not a.required, spec.essential.index(a.dest) if a.dest in spec.essential else 0))
    rest = [a for a in args if a not in essential]

    def draw(group):
        values_, flags = [a for a in group if a.kind != FLAG], [a for a in group if a.kind == FLAG]
        cols = st.columns(2)
        for i, a in enumerate(values_):
            with cols[i % 2]:
                form.render(spec.key, a, defaults.get(a.dest), options.get(a.dest))
        if flags:
            fcols = st.columns(2)
            for i, a in enumerate(flags):
                with fcols[i % 2]:
                    form.render(spec.key, a, None, None)

    st.subheader("Settings")
    st.caption("Every field starts at the value the script would use anyway. Change only what you need. "
               "Hover over ⓘ for an explanation and the config each default comes from.")
    if essential:
        draw(essential)
    if rest:
        with st.expander(f"Advanced settings ({len(rest)})"):
            draw(rest)

    values = _values(spec, args, options)
    defaults, options = resolve(spec, args, values)
    cmd = command_args(args, values, flat(args, defaults))
    changed = [a for a in args if not a.required and a.flag in cmd]

    st.subheader("Command")
    left, right = st.columns([4, 1], vertical_alignment="center")
    with left:
        if changed:
            st.caption("Changed from the defaults: " + ", ".join(f"**{form.label(a, form.doc(spec.key, a.dest))}**"
                                                              for a in changed))
        else:
            st.caption("All settings are at their defaults.")
    with right:
        st.button("Reset to defaults", on_click=form.reset, args=(spec.key,), width="stretch",
                  disabled=not changed)
    st.code(display_command(spec.script, cmd), language="bash", wrap_lines=True)

    missing = missing_required(args, values)
    running = [j for j in list_jobs(spec.script) if j.status in ("running", "starting")]
    if missing:
        st.info("Fill in the required fields (*) first: " + ", ".join(missing))
    if running:
        st.warning(f"{len(running)} job(s) of this script already running. Starting another one may compete for "
                   "the same files or GPU.")
    if st.button("Run", type="primary", icon=":material/play_arrow:", disabled=bool(missing)):
        risks = _risks(spec, values)
        if risks:
            _confirm_risky(spec, cmd, risks)
        else:
            _start(spec, cmd)

    jobs = list_jobs(spec.script)
    if jobs:
        st.subheader("Latest run of this script")
        chosen = st.session_state.get(f"{spec.key}::job")
        job = next((j for j in jobs if j.id == chosen), jobs[0])
        job_panel(job.id, key=f"{spec.key}-panel")
        if len(jobs) > 1:
            st.page_link(nav.PAGES["jobs"], label=f"All {len(jobs)} runs of this script → Jobs", icon=":material/history:")
