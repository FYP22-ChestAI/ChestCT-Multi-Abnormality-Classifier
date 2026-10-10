"""Pipeline flow diagrams (Mermaid), shared by the Home and Pipeline pages.

Streamlit renders Mermaid in strict mode (no HTML in labels) and as an image, so labels use Mermaid's own
markdown strings for bold text, and each group of steps gets its own row to keep the text readable.
"""
from __future__ import annotations

import streamlit as st

# step colours: white text on a saturated fill reads well in the light and the dark theme
CLASSES = {
    "done": "fill:#16a34a,stroke:#15803d,color:#fff",
    "partial": "fill:#d97706,stroke:#b45309,color:#fff",
    "todo": "fill:#2563eb,stroke:#1d4ed8,color:#fff",
    "running": "fill:#7c3aed,stroke:#6d28d9,color:#fff",
    "failed": "fill:#dc2626,stroke:#b91c1c,color:#fff",
    "na": "fill:#94a3b8,stroke:#64748b,color:#fff,stroke-dasharray:4 3",
}


def _label(title: str, subtitle: str | None) -> str:
    text = f"**{title}**" + (f"\n{subtitle}" if subtitle else "")
    return '["`' + text.replace('"', "'").replace("`", "'") + '`"]'


def flowchart(rows: list[tuple[str, list[tuple[str, str, str | None, str | None]]]]) -> str:
    """``rows``: (row title, [(node id, title, subtitle, class or None)]). Rows run top to bottom, the steps
    of a row left to right."""
    lines = ['%%{init: {"flowchart": {"nodeSpacing": 28, "rankSpacing": 34, "padding": 12}}}%%', "flowchart TB"]
    for n, (row_title, nodes) in enumerate(rows):
        lines += [f'  subgraph row{n}["{row_title}"]', "    direction LR"]
        prev = None
        for node_id, title, subtitle, cls in nodes:
            lines.append(f"    {node_id}{_label(title, subtitle)}" + (f":::{cls}" if cls else ""))
            if prev:
                lines.append(f"    {prev} --> {node_id}")
            prev = node_id
        lines += ["  end", f"  style row{n} fill:transparent,stroke:#94a3b8,stroke-dasharray:3 3"]
    lines += [f"  row{n} --> row{n + 1}" for n in range(len(rows) - 1)]
    lines += [f"  classDef {name} {style}" for name, style in CLASSES.items()]
    return "\n".join(lines)


def show(rows) -> None:
    st.mermaid_chart(flowchart(rows), width="content")
