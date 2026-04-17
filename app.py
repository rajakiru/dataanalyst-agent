"""
Streamlit web UI for the automated data analysis system.
"""

import io
import os
import shutil
import tempfile
import zipfile

import streamlit as st
from PIL import Image

_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
_MAX_ZIP_MB = 500


def _find_image_dir(base_dir: str) -> str:
    """Return the directory actually containing images after ZIP extraction.

    ZIPs created by compressing a folder extract as base_dir/foldername/images.
    We check one level deep so both layouts work transparently.
    """
    if any(os.path.splitext(f)[1].lower() in _IMAGE_EXTENSIONS
           for f in os.listdir(base_dir)):
        return base_dir
    for entry in os.listdir(base_dir):
        sub = os.path.join(base_dir, entry)
        if os.path.isdir(sub) and any(
            os.path.splitext(f)[1].lower() in _IMAGE_EXTENSIONS
            for f in os.listdir(sub)
        ):
            return sub
    return base_dir  # fallback — let agent produce a clear error

from agent import DEFAULT_MODEL, run_pipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AutoAnalyst — Multi-Agent Data Analysis",
    page_icon="📊",
    layout="wide",
)

def _build_markdown_report(results: dict, filename: str, selected_solutions: set | None = None) -> str:
    """Compose a self-contained markdown report from pipeline results.

    selected_solutions: set of action keys (col|issue_type|action_name) to include.
    None means include everything (pre-HITL behaviour).
    """
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

    # Solutions section
    solutions_narrative = results.get("solutions_narrative", "")
    if solutions_narrative:
        parts.append("\n## Remediation Recommendations\n")
        parts.append(solutions_narrative + "\n")
        
        solutions_tool_results = results.get("solutions_tool_results", [])
        for tool_result in solutions_tool_results:
            if tool_result["tool"] == "recommend_solutions":
                recs = tool_result["result"]
                recommendations = recs.get("recommendations", [])
                if recommendations:
                    parts.append("\n### Detailed Solutions by Issue\n")
                    for rec in recommendations:
                        col_name = rec.get("column", "—")
                        issue_type = rec.get("issue_type", "unknown")
                        severity = rec.get("severity", "medium")
                        actions = rec.get("actions", [])

                        # Filter to selected actions when caller provides a set
                        if selected_solutions is not None:
                            actions = [
                                a for a in actions
                                if f"{col_name}|{issue_type}|{a.get('action', '')}" in selected_solutions
                            ]
                        if not actions:
                            continue

                        parts.append(f"\n#### {col_name} — {issue_type} [{severity}]\n")

                        for action in actions:
                            priority = action.get("priority", "medium").upper()
                            action_name = action.get("action", "")
                            rationale = action.get("rationale", "")
                            implementation = action.get("implementation", "")

                            parts.append(f"\n**[{priority}]** {action_name}\n\n")
                            parts.append(f"_Rationale:_ {rationale}\n\n")
                            if implementation:
                                parts.append(f"```python\n{implementation}\n```\n\n")

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
    st.markdown(
        "<div style='padding: 0.5rem 0 0.25rem 0;'>"
        "<span style='font-size:1.6rem; font-weight:800; color:#1C2B4A;'>AutoAnalyst</span><br>"
        "<span style='font-size:0.8rem; color:#888; letter-spacing:0.05em;'>"
        "MULTI-AGENT CSV ANALYSIS</span></div>",
        unsafe_allow_html=True,
    )
    st.divider()

    model = st.selectbox(
        "Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0,
        help="OpenAI model used by the collection and reporting agents.",
    )

    if "results" in st.session_state and "uploaded_filename" in st.session_state:
        st.divider()
        st.caption("DOWNLOADS")
        _res = st.session_state["results"]
        _fname = st.session_state["uploaded_filename"]

        _sel_sidebar = st.session_state.get("selected_solutions") or None
        md_report = _build_markdown_report(_res, _fname, selected_solutions=_sel_sidebar)
        st.download_button(
            label="Report (.md)",
            data=md_report,
            file_name=f"{os.path.splitext(_fname)[0]}_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

        if _res.get("plot_paths"):
            zip_bytes = _build_plots_zip(_res["plot_paths"])
            st.download_button(
                label="Plots (.zip)",
                data=zip_bytes,
                file_name=f"{os.path.splitext(_fname)[0]}_plots.zip",
                mime="application/zip",
                use_container_width=True,
            )

    st.divider()
    st.caption("HOW IT WORKS")
    for step, text in [
        ("1", "Upload a CSV or ZIP of images"),
        ("2", "Add optional context so agents know your domain"),
        ("3", "Collection agent infers schema, stats, and plots"),
        ("4", "Quality agent audits issues — dismiss any that are intentional"),
        ("5", "Solutions agent recommends fixes — check the ones you want"),
        ("6", "Report exports only your selected actions"),
    ]:
        st.markdown(
            f"<div style='display:flex; gap:10px; align-items:flex-start; "
            f"margin-bottom:8px; font-size:0.85rem;'>"
            f"<span style='background:#5B8DEF; color:white; border-radius:50%; "
            f"width:20px; height:20px; display:flex; align-items:center; "
            f"justify-content:center; font-size:0.7rem; flex-shrink:0; "
            f"margin-top:1px;'>{step}</span>"
            f"<span style='color:#444; line-height:1.4;'>{text}</span></div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='font-size:2.4rem; font-weight:800; margin-bottom:0; line-height:1.2;'>"
    "AutoAnalyst</h1>"
    "<p style='font-size:1.1rem; color:#5B8DEF; font-weight:500; margin-top:4px;'>"
    "Automated Multi-Agent Data Analysis</p>"
    "<p style='font-size:0.92rem; color:#888; margin-top:2px; margin-bottom:1.5rem;'>"
    "Upload a CSV — the agents handle schema inference, statistics, quality auditing, "
    "and report writing automatically.</p>",
    unsafe_allow_html=True,
)

st.markdown("**Upload your dataset**")
st.caption("CSV file — or a ZIP archive containing images (JPEG/PNG/etc.)")
uploaded_file = st.file_uploader("", type=["csv", "zip"], label_visibility="collapsed")

if uploaded_file is not None:
    with st.expander("Analysis options", expanded=False):
        st.caption("Schema & Stats is always run. Toggle the others to control scope.")
        st.text("Schema & Stats — always on", )
        run_viz      = st.checkbox("Visualizations",         value=True, key="opt_viz")
        run_anomaly  = st.checkbox("Anomaly Detection",      value=True, key="opt_anomaly")
        run_quality  = st.checkbox("Data Quality Audit",     value=True, key="opt_quality")
        run_solutions = st.checkbox("Solutions & Remediation", value=True, key="opt_solutions")

    run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

    if run_btn:
        # Clear any previous results and HITL state so stale data never bleeds into the new run
        for key in ("results", "uploaded_filename", "dismissed_issues", "selected_solutions"):
            st.session_state.pop(key, None)

        tmp_path = None
        tmp_image_dir = None

        try:
            if uploaded_file.name.lower().endswith(".zip"):
                # Guard: reject oversized ZIPs before extracting
                zip_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                if zip_mb > _MAX_ZIP_MB:
                    st.error(f"ZIP file is {zip_mb:.0f} MB — maximum allowed is {_MAX_ZIP_MB} MB.")
                    st.stop()

                # Extract ZIP; handle nested folder layout transparently
                tmp_image_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as zf:
                    zf.extractall(tmp_image_dir)
                pipeline_input = _find_image_dir(tmp_image_dir)
            else:
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                pipeline_input = tmp_path

            enabled = frozenset(filter(None, [
                "visualizations"    if run_viz       else None,
                "anomaly_detection" if run_anomaly   else None,
                "data_quality"      if run_quality   else None,
                "solutions"         if run_solutions else None,
            ]))
            with st.spinner("Agents working... this may take a minute."):
                results = run_pipeline(pipeline_input, model=model, enabled_categories=enabled)
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
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if tmp_image_dir:
                shutil.rmtree(tmp_image_dir, ignore_errors=True)

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

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Schema & Stats", "Visualizations", "Anomalies", "Data Quality", "Solutions", "Report"])

    analysis_plan = results.get("analysis_plan", "")

    # ------------------------------------------------------------------
    # Tab 1: Schema & Statistics
    # ------------------------------------------------------------------
    with tab1:
        img_meta = results.get("image_processing_metadata")
        if img_meta:
            coverage = img_meta.get("coverage_percent", 0)
            processed = img_meta.get("processed_count", 0)
            total = img_meta.get("total_count", 0)
            failed = img_meta.get("failed_count", 0)
            feat_dim = img_meta.get("feature_dimension", 0)
            st.info(
                f"**Image dataset** — {processed}/{total} images processed "
                f"({coverage:.1f}% coverage) · {feat_dim}-dim features · "
                f"{failed} failed"
            )
            if img_meta.get("errors"):
                with st.expander("Image processing errors"):
                    for fname, err in img_meta["errors"][:10]:
                        st.caption(f"`{fname}`: {err}")
            st.divider()

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

            # Initialise dismiss state
            if "dismissed_issues" not in st.session_state:
                st.session_state["dismissed_issues"] = set()

            total_issues = len(issues) + (1 if dup_count else 0)
            if total_issues == 0:
                st.success("No issues detected — this dataset looks clean.")
            else:
                dismissed = st.session_state["dismissed_issues"]
                active_count = sum(
                    1 for i in high_issues + medium_issues
                    if f"{i['column']}|{i['issue']}" not in dismissed
                ) + (1 if dup_count and "duplicates|duplicate_rows" not in dismissed else 0)

                st.subheader(f"Issues  ({active_count} active · {len(dismissed)} dismissed)")
                st.caption("Dismiss issues you've already handled or that are intentional — the Solutions tab will skip them.")

                for issue in high_issues + medium_issues:
                    issue_key = f"{issue['column']}|{issue['issue']}"
                    is_dismissed = issue_key in dismissed
                    col_msg, col_btn = st.columns([9, 1])
                    with col_msg:
                        if is_dismissed:
                            st.caption(f"~~**{issue['column']}** — {issue['issue']}~~ _(dismissed)_")
                        elif issue["severity"] == "high":
                            st.error(f"**{issue['column']}** — {issue['issue']}")
                        else:
                            st.warning(f"**{issue['column']}** — {issue['issue']}")
                    with col_btn:
                        btn_label = "↩ Restore" if is_dismissed else "✕ Dismiss"
                        if st.button(btn_label, key=f"dismiss_{issue_key}", use_container_width=True):
                            if is_dismissed:
                                st.session_state["dismissed_issues"].discard(issue_key)
                            else:
                                st.session_state["dismissed_issues"].add(issue_key)
                            st.rerun()

                if dup_count:
                    dup_key = "duplicates|duplicate_rows"
                    is_dup_dismissed = dup_key in dismissed
                    col_msg, col_btn = st.columns([9, 1])
                    with col_msg:
                        if is_dup_dismissed:
                            st.caption(
                                f"~~**Duplicate rows** — {dup_count} rows "
                                f"({dup_result.get('duplicate_pct', 0):.1f}%) are exact duplicates.~~ _(dismissed)_"
                            )
                        else:
                            st.warning(
                                f"**Duplicate rows** — {dup_count} rows "
                                f"({dup_result.get('duplicate_pct', 0):.1f}%) are exact duplicates."
                            )
                    with col_btn:
                        btn_label = "↩ Restore" if is_dup_dismissed else "✕ Dismiss"
                        if st.button(btn_label, key=f"dismiss_{dup_key}", use_container_width=True):
                            if is_dup_dismissed:
                                st.session_state["dismissed_issues"].discard(dup_key)
                            else:
                                st.session_state["dismissed_issues"].add(dup_key)
                            st.rerun()
                    if not is_dup_dismissed:
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
    # Tab 5: Solutions & Remediation
    # ------------------------------------------------------------------
    with tab5:
        st.subheader("Solutions & Remediation Recommendations")

        solutions_narrative = results.get("solutions_narrative", "")
        solutions_tool_results = results.get("solutions_tool_results", [])

        if not solutions_narrative and not solutions_tool_results:
            st.info("No data quality issues detected — no remediation needed.")
        else:
            # Initialise selection state
            if "selected_solutions" not in st.session_state:
                st.session_state["selected_solutions"] = set()

            dismissed_issues = st.session_state.get("dismissed_issues", set())

            # Section 1: High-level remediation strategy
            if solutions_narrative:
                with st.expander("Remediation Strategy", expanded=True):
                    st.markdown(solutions_narrative)
                st.divider()

            # Section 2: Detailed per-issue solutions with checkboxes
            if solutions_tool_results:
                for tool_result in solutions_tool_results:
                    if tool_result["tool"] != "recommend_solutions":
                        continue
                    recommendations = tool_result["result"].get("recommendations", [])
                    if not recommendations:
                        continue

                    st.subheader("Actionable Solutions by Issue")
                    st.caption("Check the actions you want to implement — only checked items appear in the exported report.")

                    _widget_idx = 0
                    for rec in recommendations:
                        col_name = rec.get("column", "—")
                        issue_type = rec.get("issue_type", "unknown")
                        severity = rec.get("severity", "medium")
                        actions = rec.get("actions", [])

                        # Skip issues the user dismissed in the Quality tab
                        if any(col_name in dk for dk in dismissed_issues):
                            continue

                        severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")

                        with st.expander(f"{severity_color} {col_name} — {issue_type}"):
                            for action in actions:
                                priority = action.get("priority", "medium").upper()
                                action_name = action.get("action", "")
                                rationale = action.get("rationale", "")
                                implementation = action.get("implementation", "")

                                action_key = f"{col_name}|{issue_type}|{action_name}"
                                is_checked = action_key in st.session_state["selected_solutions"]

                                # Use a stable numeric index so keys are always unique
                                checked = st.checkbox(
                                    f"**[{priority}]** {action_name}",
                                    value=is_checked,
                                    key=f"sol_{_widget_idx}",
                                )
                                _widget_idx += 1
                                if checked:
                                    st.session_state["selected_solutions"].add(action_key)
                                else:
                                    st.session_state["selected_solutions"].discard(action_key)

                                st.caption(f"**Rationale:** {rationale}")
                                if implementation:
                                    st.code(implementation, language="python")
                                st.divider()

            sel_count = len(st.session_state["selected_solutions"])
            if sel_count:
                st.info(f"{sel_count} action(s) selected — the exported report will include only these.")

    # ------------------------------------------------------------------
    # Tab 6: Report
    # ------------------------------------------------------------------
    with tab6:
        st.subheader("Automated Analysis Report")
        fname = st.session_state.get("uploaded_filename", "analysis")
        _sel = st.session_state.get("selected_solutions") or None
        md_report = _build_markdown_report(results, fname, selected_solutions=_sel)

        if _sel:
            st.info(f"Report filtered to {len(_sel)} selected solution(s). Clear selections in the Solutions tab to export everything.")

        st.download_button(
            label="Download Report (.md)",
            data=md_report,
            file_name=f"{os.path.splitext(fname)[0]}_report.md",
            mime="text/markdown",
        )
        st.divider()
        st.markdown(narrative)
