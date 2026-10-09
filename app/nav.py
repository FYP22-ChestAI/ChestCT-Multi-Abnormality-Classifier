"""The app's pages, filled in by app/main.py, so any page can link to any other (st.page_link)."""
from __future__ import annotations

PAGES: dict = {}  # key -> st.Page: "home", "jobs", "results", and every ScriptSpec.key
