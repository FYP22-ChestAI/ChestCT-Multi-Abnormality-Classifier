"""ChestAI control panel: run every pipeline script from the browser.

    pip install -e ".[ui]"
    streamlit run app/main.py

Pages come from app/registry.py (one per script, grouped by phase) plus Home, Jobs and Results.
"""
from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)  # configs and data paths are relative to the repo root, as for the scripts
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import streamlit as st  # noqa: E402

from app import nav  # noqa: E402
from app.jobs import list_jobs  # noqa: E402
from app.registry import by_phase  # noqa: E402
from app.views.home import home_page  # noqa: E402
from app.views.jobs import jobs_page  # noqa: E402
from app.views.pipeline import pipeline_page  # noqa: E402
from app.views.results import results_page  # noqa: E402
from app.views.script import script_page  # noqa: E402

st.set_page_config(page_title="ChestAI pipeline", page_icon=":material/radiology:", layout="wide")

nav.PAGES["home"] = st.Page(home_page, title="Home", icon=":material/home:", default=True)
nav.PAGES["pipeline"] = st.Page(pipeline_page, title="Run whole pipeline", icon=":material/rocket_launch:",
                                url_path="pipeline")
nav.PAGES["jobs"] = st.Page(jobs_page, title="Jobs", icon=":material/history:", url_path="jobs")
nav.PAGES["results"] = st.Page(results_page, title="Results", icon=":material/leaderboard:", url_path="results")

sections = {"": [nav.PAGES["home"], nav.PAGES["pipeline"]]}
for phase, specs in by_phase().items():
    pages = []
    for spec in specs:
        nav.PAGES[spec.key] = st.Page(partial(script_page, spec), title=spec.title, icon=spec.icon, url_path=spec.key)
        pages.append(nav.PAGES[spec.key])
    sections[phase] = pages
sections["Monitor"] = [nav.PAGES["jobs"], nav.PAGES["results"]]

page = st.navigation(sections, expanded=True)

with st.sidebar:
    running = [j for j in list_jobs() if j.status in ("running", "starting")]
    if running:
        st.page_link(nav.PAGES["jobs"], label=f"{len(running)} job(s) running", icon=":material/autorenew:")

page.run()
