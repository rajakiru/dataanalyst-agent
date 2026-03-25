"""
Streamlit web UI for the automated data analysis system.
"""

import io
import os
import tempfile
import zipfile

import streamlit as st
from PIL import Image

from agent import DEFAULT_MODEL, run_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AutoAnalyst — Multi-Agent Data Analysis",
    page_icon="📊",
    layout="wide",
)

def _build_markdown_report(results: dict, filename: str) -> str:
    """Compose a self-contained markdown report from pipeline results."""
    parts = [f"# AutoAnalyst Report: {filename}\n"]

    plan = results.get("analysis_plan", "")
    if plan:
        parts.append("## Agent's Analysis Plan\n")
        parts.append(plan + "\n")

    tool_results = results.get("tool_results", [])
    by_tool: dict = {}
    for r in tool_results:
        by_tool.setdefault(r["tool"], []).append(r)

    if "infer_schema" in by_tool:
        schema = by_tool["infer_schema"][0]["result"]
        shape = schema.get("shape", {})
        parts.append("## Dataset Overview\n")
        parts.append(f"- **Rows:** {shape.get('rows', '?')}  ")
        parts.append(f"- **Columns:** {shape.get('cols', '?')}\n")
        roles = schema.get("roles", {})
        nulls = schema.get("null_pct", {})
        parts.append("\n| Column | Role | Null % |\n|---|---|---|\n")
        for col, role in roles.items():
            parts.append(f"| {col} | {role} | {nulls.get(col, 0)} |\n")

    if "compute_correlations" in by_tool:
        top = by_tool["compute_correlations"][0]["result"].get("top_correlations", [])
        if top:
            parts.append("\n## Top Correlations\n")
            parts.append("| Column 1 | Column 2 | Correlation |\n|---|---|---|\n")
            for p in top:
                parts.append(f"| {p['col1']} | {p['col2']} | {p['correlation']} |\n")

    tools_run = [r["tool"] for r in tool_results]
    if tools_run:
        parts.append("\n## Tools Executed\n")
        for t in tools_run:
            parts.append(f"- `{t}`\n")

    # Data quality section
    q_tool_results = results.get("quality_tool_results", [])
    q_by_tool: dict = {}
    for r in q_tool_results:
        q_by_tool.setdefault(r["tool"], []).append(r)

    if "compute_data_quality_score" in q_by_tool:
        dq = q_by_tool["compute_data_quality_score"][0]["result"]
        parts.append("\n## Data Quality Summary\n")
        parts.append(f"- **Overall Score:** {dq.get('overall_score', '?')} / 100\n")
        bd = dq.get("breakdown", {})
        parts.append(f"- **Completeness:** {bd.get('completeness', '?')}%\n")
        parts.append(f"- **Uniqueness:** {bd.get('uniqueness', '?')}%\n")
        parts.append(f"- **Consistency:** {bd.get('consistency', '?')}%\n")
        issues = dq.get("issues", [])
        if issues:
            parts.append("\n### Issues Found\n")
            parts.append("| Column | Issue | Severity |\n|---|---|---|\n")
            for issue in issues:
                parts.append(f"| {issue['column']} | {issue['issue']} | {issue['severity']} |\n")

    quality_narrative = results.get("quality_narrative", "")
    if quality_narrative:
        parts.append("\n## Quality Agent's Assessment\n")
        parts.append(quality_narrative + "\n")

    parts.append("\n## Narrative Report\n")
    parts.append(results.get("narrative", "") + "\n")

    return "\n".join(parts)


def _build_plots_zip(plot_paths: list[str]) -> bytes:
    """Pack all plot PNGs into an in-memory ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in plot_paths:
            if os.path.exists(path):
                zf.write(path, os.path.basename(path))
    return buf.getvalue()


AVAILABLE_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("AutoAnalyst")
    st.caption("Multi-Agent CSV Analysis System")
    st.divider()

    model = st.selectbox(
        "Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0,
        help="OpenAI model used by the collection and reporting agents.",
    )

    if "results" in st.session_state and "uploaded_filename" in st.session_state:
        st.divider()
        st.markdown("**Downloads**")
        _res = st.session_state["results"]
        _fname = st.session_state["uploaded_filename"]

        md_report = _build_markdown_report(_res, _fname)
        st.download_button(
            label="Download Report (.md)",
            data=md_report,
            file_name=f"{os.path.splitext(_fname)[0]}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        if _res.get("plot_paths"):
            zip_bytes = _build_plots_zip(_res["plot_paths"])
            st.download_button(
                label="Download Plots (.zip)",
                data=zip_bytes,
                file_name=f"{os.path.splitext(_fname)[0]}_plots.zip",
                mime="application/zip",
                use_container_width=True,
            )

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload a CSV\n"
        "2. The **collection agent** infers schema and autonomously calls MCP tools\n"
        "3. The **reporting agent** writes a narrative summary\n"
        "4. Results appear in the tabs below"
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("AutoAnalyst: Automated Multi-Agent Data Analysis")
st.caption("Upload a CSV — the agent handles everything else.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_btn:
        # Clear any previous results so stale data never bleeds into the new run
        for key in ("results", "uploaded_filename"):
            st.session_state.pop(key, None)

        # Save uploaded file to a temp path the MCP server can access
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("Agents working... this may take a minute."):
                results = run_pipeline(tmp_path, model=model)
        except Exception as e:
            # Recursively unwrap nested ExceptionGroups (from asyncio TaskGroup)
            def _leaf_exceptions(exc):
                causes = getattr(exc, "exceptions", None)
                if causes:
                    return [leaf for inner in causes for leaf in _leaf_exceptions(inner)]
                return [exc]
            for leaf in _leaf_exceptions(e):
                st.error(f"Pipeline failed: {type(leaf).__name__}: {leaf}")
            st.stop()
        finally:
            os.unlink(tmp_path)

        # Store results in session state so tabs can access them
        st.session_state["results"] = results
        st.session_state["uploaded_filename"] = uploaded_file.name

# Render results if available
if "results" in st.session_state:
    results = st.session_state["results"]
    tool_results = results["tool_results"]
    plot_paths = results["plot_paths"]
    narrative = results["narrative"]

    # Build lookup by tool name for easy access
    by_tool: dict = {}
    for r in tool_results:
        by_tool.setdefault(r["tool"], []).append(r)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Schema & Stats", "Visualizations", "Anomalies", "Data Quality", "Report"])

    analysis_plan = results.get("analysis_plan", "")

    # ------------------------------------------------------------------
    # Tab 1: Schema & Statistics
    # ------------------------------------------------------------------
    with tab1:
        if analysis_plan:
            with st.expander("Agent's Analysis Plan", expanded=True):
                st.markdown(analysis_plan)
            st.divider()

        col_left, col_right = st.columns(2)

        # Schema
        with col_left:
            st.subheader("Dataset Schema")
            if "infer_schema" in by_tool:
                schema = by_tool["infer_schema"][0]["result"]
                shape = schema.get("shape", {})
                st.metric("Rows", shape.get("rows", "?"))
                st.metric("Columns", shape.get("cols", "?"))

                roles = schema.get("roles", {})
                nulls = schema.get("null_pct", {})
                import pandas as pd
                schema_df = pd.DataFrame({
                    "Column": list(roles.keys()),
                    "Role": list(roles.values()),
                    "Null %": [nulls.get(c, 0) for c in roles.keys()],
                    "Dtype": [schema.get("dtypes", {}).get(c, "") for c in roles.keys()],
                })
                st.dataframe(schema_df, use_container_width=True, hide_index=True)

        # Statistics
        with col_right:
            st.subheader("Descriptive Statistics")
            if "summarize_statistics" in by_tool:
                stats = by_tool["summarize_statistics"][0]["result"].get("statistics", {})
                if stats:
                    import pandas as pd
                    stats_df = pd.DataFrame(stats).T
                    st.dataframe(stats_df.style.format("{:.3f}", na_rep="—"),
                                 use_container_width=True)

        # Correlations table
        if "compute_correlations" in by_tool:
            st.subheader("Top Correlations")
            top = by_tool["compute_correlations"][0]["result"].get("top_correlations", [])
            if top:
                import pandas as pd
                corr_df = pd.DataFrame(top)
                st.dataframe(corr_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Tab 2: Visualizations
    # ------------------------------------------------------------------
    with tab2:
        st.subheader("Generated Plots")
        if not plot_paths:
            st.info("No plots were generated.")
        else:
            zip_bytes = _build_plots_zip(plot_paths)
            fname = st.session_state.get("uploaded_filename", "analysis")
            st.download_button(
                label=f"Download all {len(plot_paths)} plots (.zip)",
                data=zip_bytes,
                file_name=f"{os.path.splitext(fname)[0]}_plots.zip",
                mime="application/zip",
            )
            st.divider()
            # Display in a 2-column grid
            cols = st.columns(2)
            for i, path in enumerate(plot_paths):
                if os.path.exists(path):
                    with cols[i % 2]:
                        img = Image.open(path)
                        st.image(img, caption=os.path.basename(path), use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 3: Anomalies
    # ------------------------------------------------------------------
    with tab3:
        st.subheader("Anomaly Detection Results")
        anomaly_entries = by_tool.get("detect_anomalies", [])
        if not anomaly_entries:
            st.info("No anomaly detection was run.")
        else:
            for entry in anomaly_entries:
                r = entry["result"]
                col = r.get("column", entry["args"].get("column", "?"))
                count = r.get("outlier_count", 0)
                pct = r.get("outlier_pct", 0)

                with st.expander(f"{col} — {count} outliers ({pct}%)"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Outlier Count", count)
                    c2.metric("Outlier %", f"{pct}%")
                    bounds = r.get("iqr_bounds", {})
                    c3.metric("IQR Bounds", f"[{bounds.get('lower','?')}, {bounds.get('upper','?')}]")

                    indices = r.get("outlier_indices", [])
                    if indices:
                        st.caption(f"Outlier row indices (first 50): {indices}")

    # ------------------------------------------------------------------
    # Tab 4: Data Quality
    # ------------------------------------------------------------------
    with tab4:
        import pandas as pd

        st.subheader("Data Quality Audit")
        q_tool_results = results.get("quality_tool_results", [])
        q_plot_paths   = results.get("quality_plot_paths", [])
        q_narrative    = results.get("quality_narrative", "")

        q_by_tool: dict = {}
        for r in q_tool_results:
            q_by_tool.setdefault(r["tool"], []).append(r)

        if "compute_data_quality_score" not in q_by_tool:
            st.info("Data quality audit did not run.")
        else:
            dq        = q_by_tool["compute_data_quality_score"][0]["result"]
            overall   = dq.get("overall_score", 0)
            breakdown = dq.get("breakdown", {})
            issues    = dq.get("issues", [])
            grade     = "Excellent" if overall >= 90 else "Good" if overall >= 75 else "Fair" if overall >= 55 else "Poor"

            # ------------------------------------------------------------------
            # Section 1: Score + dimension summary
            # ------------------------------------------------------------------
            left, right = st.columns([1, 2], gap="large")

            with left:
                st.metric("Overall Score", f"{overall} / 100", delta=grade,
                          delta_color="off")
                st.progress(int(overall) / 100)

            with right:
                for label, val, tip in [
                    ("Completeness", breakdown.get("completeness", 0), "Non-null values"),
                    ("Uniqueness",   breakdown.get("uniqueness",   0), "Non-duplicate rows"),
                    ("Consistency",  breakdown.get("consistency",  0), "No mixed types / zero-variance"),
                ]:
                    st.caption(f"**{label}** — {val:.1f}%  _{tip}_")
                    st.progress(int(val) / 100)

            st.divider()

            # ------------------------------------------------------------------
            # Section 2: Per-column health grid
            # ------------------------------------------------------------------
            st.subheader("Column Health")
            schema_result = by_tool.get("infer_schema", [{}])[0].get("result", {})
            roles     = schema_result.get("roles", {})
            null_pcts = schema_result.get("null_pct", {})
            issue_cols = {i["column"] for i in issues}

            rows = []
            for col, role in roles.items():
                null_p = null_pcts.get(col, 0)
                if null_p == 0 and col not in issue_cols:
                    status = "🟢  OK"
                elif null_p > 50 or any(
                        i["severity"] == "high" for i in issues if i["column"] == col):
                    status = "🔴  High"
                else:
                    status = "🟡  Medium"
                rows.append({
                    "Column":       col,
                    "Type":         role,
                    "Complete":     f"{100 - null_p:.1f}%",
                    "Missing":      f"{null_p:.1f}%",
                    "Status":       status,
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.divider()

            # ------------------------------------------------------------------
            # Section 3: Issues
            # ------------------------------------------------------------------
            dup_result = q_by_tool.get("detect_duplicates", [{}])[0].get("result", {})
            dup_count  = dup_result.get("duplicate_rows", 0)
            high_issues   = [i for i in issues if i["severity"] == "high"]
            medium_issues = [i for i in issues if i["severity"] == "medium"]

            total_issues = len(issues) + (1 if dup_count else 0)
            if total_issues == 0:
                st.success("No issues detected — this dataset looks clean.")
            else:
                st.subheader(f"Issues  ({total_issues} found)")
                for issue in high_issues:
                    st.error(f"**{issue['column']}** — {issue['issue']}")
                for issue in medium_issues:
                    st.warning(f"**{issue['column']}** — {issue['issue']}")
                if dup_count:
                    st.warning(
                        f"**Duplicate rows** — {dup_count} rows "
                        f"({dup_result.get('duplicate_pct', 0):.1f}%) are exact duplicates."
                    )
                    examples = dup_result.get("examples", [])
                    if examples:
                        with st.expander("Show examples"):
                            st.dataframe(pd.DataFrame(examples), use_container_width=True, hide_index=True)

        # ------------------------------------------------------------------
        # Section 5: Quality plots (missing heatmap + Q-Q plots)
        # ------------------------------------------------------------------
        if q_plot_paths:
            st.divider()

            # Separate missing heatmap from Q-Q plots
            heatmap_paths = [p for p in q_plot_paths if "missing" in os.path.basename(p)]
            qq_paths      = [p for p in q_plot_paths if "qq_" in os.path.basename(p)]

            if heatmap_paths:
                st.subheader("Missing Value Map")
                for path in heatmap_paths:
                    if os.path.exists(path):
                        st.image(Image.open(path), use_container_width=True)

            if qq_paths:
                st.subheader("Q-Q Plots (Normality Check)")
                st.caption("Points close to the red line indicate a normal distribution.")
                cols = st.columns(3)
                for i, path in enumerate(qq_paths):
                    if os.path.exists(path):
                        with cols[i % 3]:
                            st.image(Image.open(path), caption=os.path.basename(path).replace("qq_", "").replace(".png", ""), use_container_width=True)

        # ------------------------------------------------------------------
        # Section 6: Quality agent narrative
        # ------------------------------------------------------------------
        if q_narrative:
            st.divider()
            st.subheader("Quality Agent's Assessment")
            st.markdown(q_narrative)

    # ------------------------------------------------------------------
    # Tab 5: Report
    # ------------------------------------------------------------------
    with tab5:
        st.subheader("Automated Analysis Report")
        fname = st.session_state.get("uploaded_filename", "analysis")
        md_report = _build_markdown_report(results, fname)
        st.download_button(
            label="Download Report (.md)",
            data=md_report,
            file_name=f"{os.path.splitext(fname)[0]}_report.md",
            mime="text/markdown",
        )
        st.divider()
        st.markdown(narrative)
