"""Start page: the pipeline in order, one card per step, plus what is on disk right now."""
from __future__ import annotations

import streamlit as st

from app import nav
from app.choices import experiment_dirs, experiments
from app.jobs import list_jobs
from app.registry import by_phase
from app.views import flow


def _flow() -> list:
    """The main steps in order, one row per phase (generated from the registry)."""
    return [(phase, [(spec.key.replace("-", "_"), spec.title, None, None) for spec in specs if not spec.optional])
            for phase, specs in by_phase().items()]


def home_page() -> None:
    st.title("ChestAI · CT pipeline")
    st.markdown("Run every step of the project from here: from raw CT to trained, evaluated models.")

    with st.container(border=True):
        st.markdown(
            "**How to use this app**\n"
            "1. Pick a step in the sidebar (or below). Steps are listed in the order they run.\n"
            "2. Every field already holds the value the script would use. Change only what you need. "
            "Hover over **ⓘ** next to a field to see what it means and where its default comes from.\n"
            "3. The page shows the exact command. Press **Run** to start it as a background job. It keeps running "
            "if you close the browser. Follow it live on the page or under **Jobs**.\n"
            "4. Compare finished experiments under **Results**."
        )

    with st.container(border=True):
        left, right = st.columns([3, 1], vertical_alignment="center")
        left.markdown("**Run everything at once.** The pipeline page checks what is already saved and runs only "
                      "the missing steps: new seeds, new experiments, or a new run from scratch.")
        right.page_link(nav.PAGES["pipeline"], label="**Run whole pipeline →**", icon=":material/rocket_launch:")

    flow.show(_flow())

    running = [j for j in list_jobs() if j.status in ("running", "starting")]
    m = st.columns(3)
    m[0].metric("Running jobs", len(running), border=True)
    m[1].metric("Experiment configs", len(experiments({})), border=True)
    m[2].metric("Finished training runs", len(experiment_dirs({})), border=True)

    for phase, specs in by_phase().items():
        st.subheader(phase)
        main, optional = [s for s in specs if not s.optional], [s for s in specs if s.optional]
        cols = st.columns(3)
        for i, spec in enumerate(main):
            with cols[i % 3].container(border=True, height="stretch"):
                st.page_link(nav.PAGES[spec.key], label=f"**{i + 1}. {spec.title}**", icon=spec.icon)
                st.caption(spec.summary)
        if optional:
            st.caption("Optional tools: " + " · ".join(f"{s.title}" for s in optional) + " (in the sidebar)")
