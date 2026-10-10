"""Read-only comparison of finished experiments (outputs/experiments), via ct_model.training.results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ct_model.training.results import load_results, mean_std, pm

ROOT = "outputs/experiments"
VAL_METRICS = ["val_macro_auroc", "val_macro_auprc", "val_micro_auroc", "val_macro_f1"]
TEST_METRICS = ["macro_auroc", "macro_auprc", "micro_auroc", "macro_f1"]


def _stamp(root: str) -> float:
    """Changes whenever a run finishes or is evaluated, so the cache below refreshes by itself."""
    return max((p.stat().st_mtime for p in Path(root).glob("*.csv")), default=0.0)


@st.cache_data(show_spinner="Reading experiment results…")
def _load(root: str, stamp: float):
    return load_results(root)


def results_page() -> None:
    st.title("Results")
    st.caption(f"Finished experiments under `{ROOT}`. Seeds of the same configuration are averaged (mean ± std).")
    try:
        r = _load(ROOT, _stamp(ROOT))
    except FileNotFoundError as exc:
        st.info(str(exc))
        return

    compare, test, curves, per_label, runs = st.tabs(
        ["Validation", "Test evaluations", "Training curves", "Per finding", "All runs"])

    with compare:
        st.caption("Validation metrics at each run's best epoch. *setup* lists what differs from the most common config.")
        metrics = [m for m in VAL_METRICS if m in r.runs]
        table = mean_std(r.runs, ["config", "setup"], metrics).sort_values(metrics[0], ascending=False)
        st.dataframe(pm(table, metrics), width="stretch")

    with test:
        if r.test.empty:
            st.info("No evaluations yet. Run **Evaluate** on a finished training run.")
        else:
            groups = sorted(r.test["group_by"].dropna().unique()) if "group_by" in r.test else ["all"]
            group_by = st.segmented_control("Group by", groups, default="all" if "all" in groups else groups[0],
                                            help="`all` = the whole split; other choices break it down by kernel "
                                                 "class or scanner.")
            frame = r.test[r.test["group_by"] == group_by] if "group_by" in r.test else r.test
            metrics = [m for m in TEST_METRICS if m in frame]
            table = mean_std(frame, ["config", "setup", "eval", "group"], metrics).sort_values(metrics[0], ascending=False)
            st.dataframe(pm(table, metrics), width="stretch")

    with curves:
        if r.curves.empty:
            st.info("No training curves found.")
        else:
            numeric = [c for c in r.curves.columns if c not in ("epoch", "step", "seed")
                       and pd.api.types.is_numeric_dtype(r.curves[c]) and not c.startswith("val_auroc_")]
            c1, c2 = st.columns([1, 2])
            metric = c1.selectbox("Metric", numeric, index=numeric.index("val_macro_auroc") if "val_macro_auroc" in numeric else 0)
            configs = sorted(r.curves["config"].dropna().unique())
            chosen = c2.multiselect("Configurations", configs, default=configs)
            frame = r.curves[r.curves["config"].isin(chosen)]
            mean = frame.groupby(["epoch", "config"])[metric].mean().unstack("config")
            st.line_chart(mean, x_label="epoch", y_label=metric)
            st.caption("Mean over seeds. Training stops at *best epoch + patience* (early stopping), "
                       "so longer lines mean later best epochs.")
            st.dataframe(r.runs[r.runs["config"].isin(chosen)][["run_id", "best_epoch", "epochs_run", "patience",
                                                                 "max_epochs", "stopped"]],
                         hide_index=True, width="stretch")

    with per_label:
        source = st.radio("Split", ["val", "test"], horizontal=True)
        frame = r.val_labels if source == "val" else r.test_labels
        if source == "test" and not frame.empty and "group_by" in frame:
            frame = frame[frame["group_by"] == "all"]
        if frame.empty:
            st.info("Nothing to show yet.")
        else:
            metric = st.selectbox("Metric", [m for m in ("auroc", "auprc", "f1", "sensitivity", "specificity")
                                             if m in frame])
            pivot = frame.groupby(["finding", "config"])[metric].mean().unstack("config")
            st.dataframe(pivot.style.format("{:.3f}").background_gradient(axis=None, cmap="Blues"),
                         width="stretch")

    with runs:
        st.dataframe(r.runs, hide_index=True, width="stretch")
