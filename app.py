"""
Streamlit web UI for the automated data analysis system.
"""

import os
import tempfile

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

AVAILABLE_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4-6",
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
        help="LLM used by the collection and reporting agents (via Dedalus).",
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
        # Save uploaded file to a temp path the MCP server can access
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("Agents working... this may take a minute."):
                results = run_pipeline(tmp_path, model=model)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()
        finally:
            os.unlink(tmp_path)

        # Store results in session state so tabs can access them
        st.session_state["results"] = results

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

    tab1, tab2, tab3, tab4 = st.tabs(["Schema & Stats", "Visualizations", "Anomalies", "Report"])

    # ------------------------------------------------------------------
    # Tab 1: Schema & Statistics
    # ------------------------------------------------------------------
    with tab1:
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
    # Tab 4: Report
    # ------------------------------------------------------------------
    with tab4:
        st.subheader("Automated Analysis Report")
        st.markdown(narrative)
