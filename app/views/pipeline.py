"""Run the whole pipeline with one click; steps whose outputs are already saved are skipped."""
from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from app import choices, form, nav
from app.commands import script_info
from app.jobs import JOBS_DIR, list_jobs, read_job, start_job
from app.pipeline import ALWAYS, AUTO, SKIP, STAGES, Target, build_steps, will_run
from app.pipeline_runner import read_progress
from app.registry import SCRIPTS
from app.views import flow
from app.views.jobs import job_panel

RUNNER = "app/pipeline_runner.py"
_LIVE = ("running", "starting")
# how a step looks: (badge colour, icon, word)
_LOOK = {"done": ("green", ":material/check_circle:", "done"), "partial": ("orange", ":material/timelapse:", "partly done"),
         "todo": ("blue", ":material/radio_button_unchecked:", "to run"), "blocked": ("gray", ":material/schedule:", "waits for earlier steps"),
         "n/a": ("gray", ":material/remove_circle_outline:", "not needed"), "always": ("blue", ":material/refresh:", "runs every time"),
         "running": ("violet", ":material/autorenew:", "running"), "failed": ("red", ":material/error:", "failed"),
         "stopped": ("orange", ":material/stop_circle:", "stopped"), "skip": ("gray", ":material/block:", "skipped by you")}
_MODE_LABEL = {AUTO: "Auto", ALWAYS: "Always", SKIP: "Skip"}
_MODE_HELP = ("**Auto** runs the step only if its outputs are missing or out of date. **Always** redoes it even "
              "if it looks done (only steps that can be redone safely). **Skip** leaves it out.")


@st.cache_data(show_spinner="Checking what is already done…", max_entries=16)
def _plan(target_json: str, pages_json: str, modes_json: str, stamp: str):
    return build_steps(Target(**json.loads(target_json)), json.loads(pages_json), json.loads(modes_json))


def _pipeline_jobs():
    return list_jobs(RUNNER)


def _plan_file(job) -> str | None:
    return job.args[0] if job and job.args else None


def _target_form() -> Target:
    with st.container(border=True):
        st.markdown("**What should the pipeline produce?**")
        c = st.columns([1, 1.3, 1])
        sources = choices.sources({}) or ["ctrate"]
        source = c[0].selectbox("Data source", sources, key="pipe::source",
                                help=form.doc("", "source").get("help"))
        runs = choices.runs({"source": source})
        run = c[1].selectbox(
            "Run", runs, key="pipe::run", index=0 if runs else None, accept_new_options=True,
            placeholder="type a new run name",
            help="An existing run is continued (finished steps are skipped). Type a new name to plan a new CT-RATE "
                 "run from scratch: its worklist is made with the settings on the *Plan a run* page.")
        split = c[2].segmented_control(
            "Evaluate on", ["test", "val"], default="test", key="pipe::split",
            help="The split every trained model is evaluated on. Normally `test`, once all choices were made on val.")
        exps = choices.experiments({})
        experiments = st.multiselect(
            "Experiments to train", exps, default=exps, key="pipe::experiments",
            help="Experiment files in `configs/model/experiments/`. Each one is trained once per seed below. "
                 "Combinations already trained (same settings, same data) are skipped.")
        seeds = st.pills(
            "Seeds", list(range(10)), selection_mode="multi", default=[0], key="pipe::seeds",
            help="Each seed is one more training run of the same configuration, to average out randomness. "
                 "Results pages show mean ± std over seeds.")
    return Target(source=source, run=run or "", experiments=list(experiments), seeds=sorted(seeds or []),
                  eval_split=split or "test")


def _page_values() -> dict[str, dict]:
    out = {}
    for spec in SCRIPTS:
        try:
            edited = form.edited_values(spec.key, script_info(spec.script)[1])
        except Exception:
            edited = {}
        if edited:
            out[spec.key] = edited
    return out


def _display_state(step, modes, live: dict) -> str:
    if step.key in live and live[step.key]["state"] in ("running", "failed", "stopped"):
        return live[step.key]["state"]
    if modes.get(step.key) == SKIP:
        return "skip"
    if modes.get(step.key) == ALWAYS and step.can_always:
        return "todo"
    return step.status.state


def _stage_class(states: list[str]) -> str:
    if "running" in states:
        return "running"
    if "failed" in states:
        return "failed"
    if all(x in ("n/a", "skip") for x in states):
        return "na"
    if all(x in ("done", "n/a", "skip") for x in states):
        return "done"
    return "partial" if "done" in states else "todo"


def _diagram(steps, modes, live) -> list:
    rows: dict[str, list] = {}
    for stage in STAGES:
        group = [s for s in steps if s.stage == stage.title]
        if not group:
            continue
        states = [_display_state(s, modes, live) for s in group]
        n_done = states.count("done")
        text = f"{n_done} / {len(group)} done" if len(group) > 1 else _LOOK.get(states[0], ("", "", states[0]))[2]
        rows.setdefault(stage.group, []).append((stage.key, stage.title, text, _stage_class(states)))
    return list(rows.items())


def _step_row(step, modes, live, title: str | None = None) -> None:
    state = _display_state(step, modes, live)
    color, icon, word = _LOOK.get(state, ("gray", ":material/help:", state))
    detail = live.get(step.key, {}).get("detail") if state == "failed" else step.status.detail
    with st.container(border=True):
        c = st.columns([6, 2.6, 1.4], vertical_alignment="center")
        with c[0]:
            st.markdown(f"**{title or step.title}** &nbsp; :{color}-badge[{icon} {word}]")
            st.caption(detail + (" · checked again when reached" if step.recheck else ""))
        with c[1]:
            options = [AUTO, ALWAYS, SKIP] if step.can_always else [AUTO, SKIP]
            st.segmented_control("Mode", options, default=AUTO, key=f"pipe::mode::{step.key}",
                                 format_func=_MODE_LABEL.get, label_visibility="collapsed", help=_MODE_HELP,
                                 disabled=step.status.state == "n/a")
        with c[2]:
            with st.popover("Command", icon=":material/terminal:", width="stretch"):
                st.code(step.command, language="bash", wrap_lines=True)
                st.page_link(nav.PAGES[step.script_key], label="Change this step's settings", icon=":material/tune:")


@st.dialog("Run the pipeline?", width="large")
def _confirm(target: Target, pages: dict, modes: dict, to_run) -> None:
    st.markdown(f"These **{len(to_run)} step(s)** will run in order, as one background job. Everything else is "
                "already done and is skipped. Each step is checked again just before it starts.")
    long = [s for s in to_run if s.long]
    if long:
        st.warning("Long-running steps (minutes to hours): " + ", ".join(f"{s.stage} · {s.title}" for s in long))
    for s in to_run:
        st.markdown(f"- **{s.stage}** · {s.title}  \n  `{s.command}`")
    if pages:
        st.info("Using the settings you changed on these steps' pages: " + ", ".join(sorted(pages)))
    st.caption("The first failure stops the pipeline. Press Run again after fixing it: finished steps are skipped.")
    if st.button("Start", type="primary", icon=":material/play_arrow:", width="stretch"):
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        plan = JOBS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_pipeline.plan.json"
        plan.write_text(json.dumps({"target": target.__dict__, "page_values": pages, "modes": modes}, indent=1,
                                   default=str), encoding="utf-8")
        job = start_job(RUNNER, [plan.relative_to(JOBS_DIR.parents[1]).as_posix()])
        st.session_state["pipe::job"] = job.id
        st.rerun()


def _live_progress(job_id: str) -> None:
    job = read_job(job_id)
    plan_file = _plan_file(job)
    running = job.status in _LIVE

    @st.fragment(run_every=2 if running else None)
    def panel():
        j = read_job(job_id)
        prog = read_progress(plan_file) if plan_file else {}
        steps = [prog["steps"][k] for k in prog.get("order", []) if k in prog.get("steps", {})]
        planned = [s for s in steps if s["state"] != "skipped"]
        finished = [s for s in planned if s["state"] in ("done", "failed", "stopped")]
        if planned:
            st.progress(len(finished) / len(planned),
                        text=f"{len(finished)} of {len(planned)} planned step(s) finished · "
                             f"{len(steps) - len(planned)} skipped (already done)")
        for s in planned:
            state = {"done": "complete", "failed": "error", "stopped": "error"}.get(s["state"], "running")
            label = f"{s['stage']} · {s['title']}"
            if s["state"] == "pending":
                st.caption(f":material/schedule: {label} · waiting")
                continue
            with st.status(label + (f" (exit code {s['exit_code']})" if s.get("exit_code") not in (None, 0) else ""),
                           state=state, expanded=False):
                st.code(s["command"], language="bash", wrap_lines=True)
        if running and j.status not in _LIVE:
            st.session_state["pipe::stamp"] = st.session_state.get("pipe::stamp", 0) + 1
            st.rerun()

    panel()
    job_panel(job_id, key="pipeline-job")


def pipeline_page() -> None:
    st.title("Run the whole pipeline")
    st.markdown("From raw CT to trained, evaluated and summarized models, in one click. Every step is checked "
                "against what is **already saved on disk** (worklist, cache, manifest, QC, splits, embeddings, "
                "trained runs, evaluations), and only what is missing runs.")

    target = _target_form()
    if not target.run:
        st.info("Choose a run, or type a new run name.")
        return
    if not target.experiments or not target.seeds:
        st.info("Choose at least one experiment and one seed.")
        return

    pages = _page_values()
    jobs = _pipeline_jobs()
    latest = next((j for j in jobs if j.id == st.session_state.get("pipe::job")), jobs[0] if jobs else None)
    busy = latest is not None and latest.status in _LIVE
    live = read_progress(_plan_file(latest)).get("steps", {}) if busy and _plan_file(latest) else {}

    modes = {k.removeprefix("pipe::mode::"): v for k, v in st.session_state.items()
             if isinstance(k, str) and k.startswith("pipe::mode::") and v and v != AUTO}
    stamp = f"{st.session_state.get('pipe::stamp', 0)}|{latest.id if latest else ''}|{latest.status if latest else ''}"
    steps = _plan(json.dumps(target.__dict__), json.dumps(pages, sort_keys=True, default=str),
                  json.dumps(modes, sort_keys=True), stamp)
    to_run = [s for s in steps if will_run(s, modes)]

    top = st.columns([1, 1, 1, 1], vertical_alignment="center")
    top[0].metric("Steps to run", len(to_run), border=True)
    top[1].metric("Already done", sum(s.status.state == "done" and s not in to_run for s in steps), border=True)
    top[2].metric("Total steps", len(steps), border=True)
    if top[3].button("Refresh status", icon=":material/refresh:", width="stretch",
                     help="Check the outputs on disk again (e.g. after running a step from its own page)."):
        st.session_state["pipe::stamp"] = st.session_state.get("pipe::stamp", 0) + 1
        st.rerun()

    flow.show(_diagram(steps, modes, live))
    if pages:
        st.caption("Using settings you changed on step pages: "
                   + ", ".join(f"**{next(s.title for s in SCRIPTS if s.key == k)}** ({', '.join(v)})"
                               for k, v in pages.items()))

    st.subheader("Steps")
    for stage in STAGES:
        group = [s for s in steps if s.stage == stage.title]
        if len(group) == 1:  # a one-step stage: name the stage, plus what the step is about (e.g. its encoder)
            only = group[0]
            _step_row(only, modes, live, only.stage if only.title == only.stage else f"{only.stage} · {only.title}")
        elif group:
            n_run = sum(s in to_run for s in group)
            n_done = sum(s.status.state == "done" for s in group)
            with st.expander(f"**{stage.title}**: {len(group)} steps · {n_done} done · {n_run} to run",
                             expanded=n_run > 0 and n_run < len(group)):
                for s in group:
                    _step_row(s, modes, live)

    st.subheader("Run")
    if busy:
        st.info("A pipeline is running. Its progress is below.")
    label = f"Run pipeline: {len(to_run)} step(s)" if to_run else "Nothing to run: everything is done"
    if st.button(label, type="primary", icon=":material/rocket_launch:", disabled=busy or not to_run):
        _confirm(target, pages, modes, to_run)

    if latest:
        st.subheader("Current pipeline run" if busy else "Last pipeline run")
        _live_progress(latest.id)
        if len(jobs) > 1:
            st.page_link(nav.PAGES["jobs"], label=f"All {len(jobs)} pipeline runs → Jobs", icon=":material/history:")
