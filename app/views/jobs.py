"""Jobs started from the UI: status, live log and Stop. The panel is shared with every script page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.jobs import JOBS_DIR, duration, list_jobs, read_job, stop_job, tail

_BADGE = {"running": ("blue", ":material/autorenew:"), "starting": ("blue", ":material/hourglass_top:"),
          "finished": ("green", ":material/check_circle:"), "failed": ("red", ":material/error:"),
          "stopped": ("orange", ":material/stop_circle:"), "lost": ("gray", ":material/help:")}
_LIVE = ("running", "starting")


def job_panel(job_id: str, key: str) -> None:
    job = read_job(job_id)
    if job is None:
        st.info("This job's record is gone.")
        return
    live = job.status in _LIVE

    @st.fragment(run_every=2 if live else None)
    def panel():
        j = read_job(job_id)
        color, icon = _BADGE[j.status]
        top = st.columns([3, 1], vertical_alignment="center")
        with top[0]:
            st.badge(f"{j.status} · {duration(j)}" + (f" · exit code {j.exit_code}" if j.exit_code is not None else ""),
                     color=color, icon=icon)
            st.caption(f"Job `{j.id}` · started {j.started.replace('T', ' ')} · log `{j.log.relative_to(JOBS_DIR.parents[1])}`")
        with top[1]:
            if j.status in _LIVE and st.button("Stop", key=f"{key}-stop", icon=":material/stop:",
                                                width="stretch"):
                stop_job(j)
                st.toast("Stop sent. The script finishes its current step and exits.")
        with st.container(height=420, autoscroll=True):  # follows the newest output
            st.code(tail(j) or "(no output yet)", language=None, wrap_lines=True)
        if live and j.status not in _LIVE:
            st.rerun()  # finished: redraw the page once, without polling any more

    panel()


def jobs_page() -> None:
    st.title("Jobs")
    st.caption("Everything started from this app. Jobs run in the background on the server: closing the browser "
               "does not stop them. Logs are kept in `outputs/ui_jobs/`.")
    jobs = list_jobs()
    if not jobs:
        st.info("No jobs yet. Open a step in the sidebar and press **Run**.")
        return
    only_live = st.toggle("Only running jobs", value=False)
    shown = [j for j in jobs if j.status in _LIVE] if only_live else jobs
    if not shown:
        st.info("Nothing is running.")
        return
    table = pd.DataFrame([{"job": j.id, "status": j.status, "script": j.script, "started": j.started.replace("T", " "),
                           "duration": duration(j), "command": j.command} for j in shown])
    event = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                         selection_mode="single-row", key="jobs-table")
    rows = event.selection.rows if event and event.selection else []
    job = shown[rows[0]] if rows else shown[0]
    st.subheader(f"{job.id}")
    st.code(job.command, language="bash", wrap_lines=True)
    job_panel(job.id, key="jobs-page")
