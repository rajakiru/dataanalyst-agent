"""
Streamlit web UI for AutoAnalyst — Workflow Builder style.
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
    return base_dir


from agent import DEFAULT_MODEL, run_plan_only, run_execution_phase

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AutoAnalyst",
    page_icon=None,
    layout="wide",
)

st.markdown("""
<style>
/* ---- Font ---- */
html, body, [class*="css"], .stMarkdown, .stText, .stMetric,
button, input, select, textarea, .stDataFrame, .stCaption,
h1, h2, h3, h4, h5, h6, p, span, div {
    font-family: 'Calibri', 'Gill Sans', 'Trebuchet MS', 'Helvetica Neue', sans-serif !important;
}

/* ---- Workflow node cards ---- */
.wf-node {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 14px 16px;
    background: white;
    border: 2px solid #E2E8F0;
    border-radius: 12px;
    min-height: 68px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: border-color 0.15s, box-shadow 0.15s;
}
.wf-node.wf-selected {
    border-color: #5B8DEF;
    box-shadow: 0 0 0 3px rgba(91,141,239,0.12), 0 1px 4px rgba(0,0,0,0.06);
    background: #F8FAFF;
}
.wf-node.wf-running  { border-color: #F59E0B; }
.wf-node.wf-complete { border-color: #10B981; }
.wf-node.wf-error    { border-color: #EF4444; }
.wf-node.wf-idle     { opacity: 0.5; }

.wf-icon {
    width: 36px; height: 36px;
    border-radius: 8px;
    background: #EEF2F8;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.72rem; font-weight: 800; color: #5B8DEF;
    flex-shrink: 0; letter-spacing: 0.02em;
}
.wf-node.wf-selected .wf-icon  { background: #5B8DEF; color: white; }
.wf-node.wf-complete .wf-icon  { background: #D1FAE5; color: #065F46; }
.wf-node.wf-running  .wf-icon  { background: #FEF3C7; color: #92400E; }

.wf-text   { flex: 1; min-width: 0; }
.wf-title  { display: block; font-weight: 600; font-size: 0.9rem; color: #1C2B4A;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wf-sub    { display: block; font-size: 0.75rem; color: #94A3B8; margin-top: 2px;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

.wf-badge {
    font-size: 0.65rem; padding: 2px 9px; border-radius: 20px; font-weight: 600;
    background: #F1F5F9; color: #64748B; flex-shrink: 0; white-space: nowrap;
}
.wf-badge.wf-complete { background: #D1FAE5; color: #065F46; }
.wf-badge.wf-running  { background: #FEF3C7; color: #92400E; animation: blink 1.2s ease-in-out infinite; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.45} }

/* Connector between nodes */
.wf-line { width: 2px; height: 20px; background: #E2E8F0; margin: 0 24px; }

/* ---- Invisible overlay button on each node card ----
   Uses CSS :has() — supported in Chrome 105+, Safari 15.4+, Firefox 121+ */
.element-container:has(.wf-click-marker) + .element-container .stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: transparent !important;
    margin-top: -70px !important;
    height: 70px !important;
    cursor: pointer !important;
    position: relative !important;
    z-index: 50 !important;
    border-radius: 12px !important;
    font-size: 0 !important;
    transition: background 0.12s !important;
}
.element-container:has(.wf-click-marker) + .element-container .stButton > button:hover {
    background: rgba(91,141,239,0.06) !important;
    border: none !important;
    box-shadow: none !important;
}
.element-container:has(.wf-click-marker) + .element-container .stButton > button:focus {
    outline: none !important;
    box-shadow: none !important;
}
/* Idle nodes: not clickable */
.element-container:has(.wf-click-idle) + .element-container .stButton > button {
    cursor: default !important;
    pointer-events: none !important;
}

/* ---- General UI rounding ---- */
.stButton > button, .stDownloadButton > button { border-radius: 10px !important; }
[data-testid="stExpander"]        { border-radius: 12px !important; }
[data-testid="stAlert"],
[data-testid="stInfo"],
[data-testid="stSuccess"],
[data-testid="stWarning"],
[data-testid="stError"]           { border-radius: 10px !important; }
[data-testid="metric-container"]  { border-radius: 12px !important; background: #F8FAFF; padding: 0.6rem 1rem !important; }
[data-testid="stDataFrame"] > div { border-radius: 10px !important; overflow: hidden; }
[data-baseweb="select"] > div     { border-radius: 10px !important; }
[data-testid="stStatusWidget"] > div { border-radius: 12px !important; }
[data-testid="stFileUploader"]    { border-radius: 10px !important; }
[data-testid="stProgressBar"] > div > div { border-radius: 6px !important; }
[data-baseweb="tab-list"]         { border-radius: 10px !important; }

/* Remove default Streamlit max-width on main block */
.block-container { max-width: 100% !important; padding-top: 1.5rem !important; }

/* Divider in header */
.hdr-divider { border: none; border-top: 1px solid #E8EDF4; margin: 8px 0 16px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_markdown_report(results: dict, filename: str, selected_solutions: set | None = None) -> str:
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
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in plot_paths:
            if os.path.exists(path):
                zf.write(path, os.path.basename(path))
    return buf.getvalue()


AVAILABLE_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_DATASETS = {
    "Iris (clean)":        os.path.join(_DATA_DIR, "iris.csv"),
    "Iris (corrupted)":    os.path.join(_DATA_DIR, "iris_corrupted.csv"),
    "Titanic":             os.path.join(os.path.dirname(__file__), "Titanic-Dataset.csv"),
    "Titanic (corrupted)": os.path.join(_DATA_DIR, "titanic_corrupted.csv"),
}

PHASE_LABELS = {
    "mcp_init":          "MCP server started",
    "schema":            "Inferring schema",
    "planning":          "Planning analysis",
    "collection":        "Running collection agent",
    "quality_score":     "Quality audit — scoring",
    "quality_tools":     "Quality audit — checks",
    "quality_narrative": "Quality agent writing report",
    "solutions":         "Solutions agent",
    "reporting":         "Reporting agent",
}

_ALWAYS_ON_TOOLS = {
    "compute_data_quality_score", "detect_duplicates",
    "plot_missing_heatmap", "plot_qq", "recommend_solutions", "infer_schema",
}


def _fmt_tool(tool: str, args: dict) -> str:
    col = args.get("column") or args.get("numeric_column") or args.get("x_column")
    return f"{tool}({col})" if col else tool


def _fmt_result(tool: str, result: dict) -> str:
    if "error" in result:
        return f"error: {str(result['error'])[:60]}"
    if tool == "infer_schema":
        s = result.get("shape", {})
        return f"{s.get('rows','?')} rows x {s.get('cols','?')} cols"
    if tool == "summarize_statistics":
        return f"stats for {len(result.get('statistics', {}))} column(s)"
    if tool == "compute_correlations":
        top = result.get("top_correlations", [])
        if top:
            t = top[0]
            return f"top: {t['col1']}/{t['col2']} r={t['correlation']}"
        return "no correlations"
    if tool == "detect_anomalies":
        return f"{result.get('outlier_count', 0)} outliers ({result.get('outlier_pct', 0)}%)"
    if tool == "compute_data_quality_score":
        return f"score: {result.get('overall_score', '?')} / 100"
    if tool == "detect_duplicates":
        return f"{result.get('duplicate_rows', 0)} duplicate rows"
    if "plot_path" in result:
        return f"saved {os.path.basename(result['plot_path'])}"
    return "done"


class TimelineRenderer:
    def __init__(self, status):
        self._status = status
        self._phase_slots: dict = {}
        self._tool_slots: dict = {}
        self._active_phase: str | None = None
        self._tool_counts: dict = {}

    def on_event(self, event: dict):
        t = event["type"]
        if t == "phase_start":
            if self._active_phase:
                self._close_phase(self._active_phase)
            phase = event["phase"]
            self._active_phase = phase
            self._tool_counts[phase] = 0
            label = PHASE_LABELS.get(phase, phase)
            slot = self._status.empty()
            self._phase_slots[phase] = slot
            slot.markdown(f"**{label}...**")
        elif t == "tool_start":
            phase = self._active_phase
            idx = self._tool_counts.get(phase, 0)
            slot = self._status.empty()
            self._tool_slots[(phase, idx)] = slot
            slot.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{_fmt_tool(event['tool'], event.get('args', {}))}`")
        elif t == "tool_result":
            phase = self._active_phase
            idx = self._tool_counts.get(phase, 0)
            slot = self._tool_slots.get((phase, idx))
            if slot:
                slot.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;`{_fmt_tool(event['tool'], {})}` — {_fmt_result(event['tool'], event['result'])}")
            self._tool_counts[phase] = idx + 1
        elif t == "plan_ready":
            slot = self._phase_slots.get("planning")
            if slot:
                slot.markdown("**Planning complete**")

    def _close_phase(self, phase: str):
        slot = self._phase_slots.get(phase)
        if not slot:
            return
        label = PHASE_LABELS.get(phase, phase)
        n = self._tool_counts.get(phase, 0)
        suffix = f" ({n} tools)" if n else ""
        slot.markdown(f"**{label}{suffix} — done**")

    def finalize(self):
        if self._active_phase:
            self._close_phase(self._active_phase)


# ---------------------------------------------------------------------------
# Workflow node definitions
# ---------------------------------------------------------------------------
NODES = [
    {"id": "trigger",   "title": "Dataset Uploaded",  "subtitle": "Trigger when CSV is uploaded", "abbr": "DB"},
    {"id": "analysis",  "title": "Analysis Agent",    "subtitle": "Schema + stats + anomalies",    "abbr": "AI"},
    {"id": "quality",   "title": "Detect Issues",     "subtitle": "Find data cascade risks",       "abbr": "QA"},
    {"id": "solutions", "title": "Suggest Fixes",     "subtitle": "Impute, transform, clean",      "abbr": "FX"},
    {"id": "report",    "title": "Generate Report",   "subtitle": "Export findings",               "abbr": "RP"},
]


def _get_node_states() -> dict:
    step = st.session_state.get("step", "upload")
    if step == "upload":
        return {n["id"]: "idle" for n in NODES}
    elif step == "review_plan":
        return {"trigger": "complete", "analysis": "idle", "quality": "idle", "solutions": "idle", "report": "idle"}
    elif step == "running":
        return {"trigger": "complete", "analysis": "running", "quality": "idle", "solutions": "idle", "report": "idle"}
    else:
        return {n["id"]: "complete" for n in NODES}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------
def _leaf_exceptions(exc):
    causes = getattr(exc, "exceptions", None)
    if causes:
        return [leaf for inner in causes for leaf in _leaf_exceptions(inner)]
    return [exc]


def _reset():
    for key in ("step", "plan_state", "results", "uploaded_filename",
                "dismissed_issues", "selected_solutions", "applied_solutions",
                "original_csv_bytes", "csv_path", "tmp_image_dir",
                "enabled_categories", "selected_node", "use_sample"):
        st.session_state.pop(key, None)


# Initialize
if "step" not in st.session_state:
    st.session_state["step"] = "upload"
if "selected_node" not in st.session_state:
    st.session_state["selected_node"] = "trigger"

# ---------------------------------------------------------------------------
# Header bar
# ---------------------------------------------------------------------------
hdr_left, hdr_mid, hdr_right = st.columns([5, 2, 2])
with hdr_left:
    st.markdown(
        "<div style='padding:2px 0 4px 0'>"
        "<span style='font-size:1.4rem;font-weight:800;color:#1C2B4A;letter-spacing:-0.01em'>AutoAnalyst</span>"
        "<span style='font-size:0.78rem;color:#94A3B8;margin-left:10px;font-weight:400'>"
        "Multi-Agent Data Analysis</span></div>",
        unsafe_allow_html=True,
    )
with hdr_mid:
    model = st.selectbox(
        "Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in AVAILABLE_MODELS else 0,
        label_visibility="collapsed",
    )
with hdr_right:
    _step_now = st.session_state["step"]
    if _step_now == "complete":
        if st.button("New Analysis", type="primary", use_container_width=True):
            _reset()
            st.rerun()
    else:
        st.markdown(
            "<div style='text-align:right;padding-top:6px'>"
            "<span style='background:#F1F5F9;color:#64748B;padding:4px 14px;"
            "border-radius:20px;font-size:0.78rem;font-weight:600'>Draft</span></div>",
            unsafe_allow_html=True,
        )

st.markdown('<hr class="hdr-divider"/>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([4, 7], gap="large")
node_states = _get_node_states()

# ---------------------------------------------------------------------------
# LEFT: Pipeline
# ---------------------------------------------------------------------------
with left_col:
    st.markdown(
        "<div style='font-size:0.68rem;font-weight:700;color:#94A3B8;"
        "letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px'>Pipeline</div>",
        unsafe_allow_html=True,
    )

    for i, node in enumerate(NODES):
        nid    = node["id"]
        state  = node_states.get(nid, "idle")
        is_sel = st.session_state["selected_node"] == nid

        # Connector line
        if i > 0:
            st.markdown('<div class="wf-line"></div>', unsafe_allow_html=True)

        # Determine card CSS class
        if is_sel:
            card_cls = "wf-selected"
        elif state in ("running", "complete", "error"):
            card_cls = f"wf-{state}"
        else:
            card_cls = "wf-idle"

        badge_text = {"complete": "done", "running": "running", "error": "error"}.get(state, "")
        badge_cls  = f"wf-badge wf-{state}" if badge_text else "wf-badge"

        # Visual card (HTML)
        st.markdown(f"""
<div class="wf-node {card_cls}">
  <div class="wf-icon">{node['abbr']}</div>
  <div class="wf-text">
    <span class="wf-title">{node['title']}</span>
    <span class="wf-sub">{node['subtitle']}</span>
  </div>
  <div class="{badge_cls}">{badge_text}</div>
</div>""", unsafe_allow_html=True)

        # CSS marker + invisible click button
        idle_cls = " wf-click-idle" if (state == "idle" and nid != "trigger" and _step_now != "complete") else ""
        st.markdown(f'<div class="wf-click-marker{idle_cls}"></div>', unsafe_allow_html=True)
        if st.button(
            nid,
            key=f"node_{nid}",
            use_container_width=True,
            disabled=(state == "idle" and nid != "trigger" and _step_now != "complete"),
        ):
            st.session_state["selected_node"] = nid
            st.rerun()

    st.markdown(
        "<div style='text-align:center;margin-top:14px;padding:10px;"
        "border:2px dashed #E2E8F0;border-radius:10px;"
        "color:#94A3B8;font-size:0.82rem'>+ Add Step</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# RIGHT: Panel
# ---------------------------------------------------------------------------
with right_col:
    step     = st.session_state["step"]
    selected = st.session_state["selected_node"]

    # =========================================================================
    # UPLOAD
    # =========================================================================
    if step == "upload":
        st.markdown("**Trigger Settings**")
        st.caption("Upload a dataset to start the pipeline.")

        uploaded_file = st.file_uploader(
            "Upload CSV or ZIP", type=["csv", "zip"], label_visibility="collapsed"
        )

        st.caption("Or try a sample dataset:")
        sample_cols = st.columns(len(SAMPLE_DATASETS))
        for idx, (label, path) in enumerate(SAMPLE_DATASETS.items()):
            file_exists = os.path.exists(path)
            if sample_cols[idx].button(
                label, use_container_width=True, disabled=not file_exists,
                help=None if file_exists else "Run data/generate_corrupted.py to create this dataset",
            ):
                _reset()
                st.session_state["csv_path"] = path
                st.session_state["uploaded_filename"] = os.path.basename(path)
                st.session_state["use_sample"] = True
                uploaded_file = None

        if uploaded_file is not None or st.session_state.get("use_sample"):
            st.divider()
            with st.expander("Analysis options", expanded=False):
                st.caption("Schema & Stats always runs. Toggle others to control scope.")
                st.text("Schema & Stats — always on")
                run_viz       = st.checkbox("Visualizations",           value=True, key="opt_viz")
                run_anomaly   = st.checkbox("Anomaly Detection",        value=True, key="opt_anomaly")
                run_quality   = st.checkbox("Data Quality Audit",       value=True, key="opt_quality")
                run_solutions = st.checkbox("Solutions & Remediation",  value=True, key="opt_solutions")

            if st.button("Run Workflow", type="primary", use_container_width=True):
                for key in ("plan_state", "results", "dismissed_issues",
                            "selected_solutions", "applied_solutions"):
                    st.session_state.pop(key, None)

                enabled = frozenset(filter(None, [
                    "visualizations"    if run_viz       else None,
                    "anomaly_detection" if run_anomaly   else None,
                    "data_quality"      if run_quality   else None,
                    "solutions"         if run_solutions else None,
                ]))
                st.session_state["enabled_categories"] = enabled

                if not st.session_state.get("use_sample"):
                    tmp_image_dir = None
                    if uploaded_file.name.lower().endswith(".zip"):
                        zip_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                        if zip_mb > _MAX_ZIP_MB:
                            st.error(f"ZIP file is {zip_mb:.0f} MB — max is {_MAX_ZIP_MB} MB.")
                            st.stop()
                        tmp_image_dir = tempfile.mkdtemp()
                        with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as zf:
                            zf.extractall(tmp_image_dir)
                        csv_path = _find_image_dir(tmp_image_dir)
                        st.session_state["tmp_image_dir"] = tmp_image_dir
                    else:
                        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            csv_path = tmp.name
                    st.session_state["csv_path"] = csv_path
                    st.session_state["uploaded_filename"] = uploaded_file.name

                st.session_state.pop("use_sample", None)

                try:
                    with st.status("Planning analysis...", expanded=True) as _status:
                        _renderer = TimelineRenderer(_status)
                        plan_state = run_plan_only(
                            st.session_state["csv_path"],
                            model=model,
                            enabled_categories=enabled,
                            on_event=_renderer.on_event,
                        )
                        _renderer.finalize()
                        _status.update(label="Plan ready", state="complete", expanded=False)
                except Exception as e:
                    for leaf in _leaf_exceptions(e):
                        st.error(f"Planning failed: {type(leaf).__name__}: {leaf}")
                    st.stop()

                st.session_state["plan_state"] = plan_state
                st.session_state["step"] = "review_plan"
                st.session_state["selected_node"] = "analysis"
                st.rerun()

    # =========================================================================
    # REVIEW PLAN
    # =========================================================================
    elif step == "review_plan":
        plan_state       = st.session_state["plan_state"]
        analysis_plan    = plan_state["analysis_plan"]
        available_tools  = [t for t in plan_state["available_tool_names"] if t not in _ALWAYS_ON_TOOLS]
        planned_tools    = [t for t in plan_state.get("planned_tool_names", []) if t not in _ALWAYS_ON_TOOLS]
        tool_descriptions = plan_state.get("tool_descriptions", {})

        st.markdown("**Analysis Node — Review Plan**")

        with st.expander("Agent's Analysis Plan", expanded=True):
            st.markdown(analysis_plan)

        st.divider()
        st.caption("Pre-checked based on the agent's plan. Uncheck tools to skip them. Quality audit always runs.")

        approved = []
        tool_cols = st.columns(2)
        for i, tool in enumerate(available_tools):
            default    = tool in planned_tools
            desc       = tool_descriptions.get(tool, "")
            short_desc = desc.split(".")[0] if desc else ""
            checked = tool_cols[i % 2].checkbox(
                f"**{tool}**", value=default, key=f"tool_{tool}",
                help=desc if desc else None,
            )
            if short_desc:
                tool_cols[i % 2].caption(f"  {short_desc}")
            if checked:
                approved.append(tool)

        st.divider()
        c1, c2 = st.columns([3, 1])
        run_clicked = c1.button(
            "Run Analysis", type="primary", use_container_width=True, disabled=len(approved) == 0
        )
        if c2.button("Start over", use_container_width=True):
            _reset()
            st.rerun()

        if run_clicked:
            st.session_state["approved_tools"] = approved
            st.session_state["step"] = "running"
            st.rerun()

    # =========================================================================
    # RUNNING
    # =========================================================================
    elif step == "running":
        plan_state     = st.session_state["plan_state"]
        approved_tools = st.session_state["approved_tools"]

        # Save CSV bytes before temp file is deleted
        _csv_before = st.session_state.get("csv_path", "")
        if _csv_before and os.path.exists(_csv_before) and _csv_before.endswith(".csv"):
            try:
                with open(_csv_before, "rb") as _fh:
                    st.session_state["original_csv_bytes"] = _fh.read()
            except OSError:
                pass

        try:
            with st.status("Running analysis...", expanded=True) as _status:
                _renderer = TimelineRenderer(_status)
                results = run_execution_phase(
                    plan_state, approved_tools, model=model,
                    on_event=_renderer.on_event,
                )
                _renderer.finalize()
                _status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as e:
            for leaf in _leaf_exceptions(e):
                st.error(f"Pipeline failed: {type(leaf).__name__}: {leaf}")
            st.stop()
        finally:
            _csv = st.session_state.get("csv_path", "")
            if _csv and os.path.exists(_csv) and _csv not in SAMPLE_DATASETS.values():
                try:
                    os.unlink(_csv)
                except OSError:
                    pass
            _tmp_dir = st.session_state.pop("tmp_image_dir", None)
            if _tmp_dir:
                shutil.rmtree(_tmp_dir, ignore_errors=True)

        st.session_state["results"] = results
        st.session_state["step"] = "complete"
        st.session_state["selected_node"] = "analysis"
        st.rerun()

    # =========================================================================
    # COMPLETE
    # =========================================================================
    elif step == "complete" and "results" in st.session_state:
        import pandas as pd
        results     = st.session_state["results"]
        tool_results = results["tool_results"]
        plot_paths   = results["plot_paths"]
        narrative    = results["narrative"]

        by_tool: dict = {}
        for r in tool_results:
            by_tool.setdefault(r["tool"], []).append(r)

        # ----------------------------------------------------------------
        # TRIGGER — dataset overview
        # ----------------------------------------------------------------
        if selected == "trigger":
            st.markdown("**Dataset**")
            fname = st.session_state.get("uploaded_filename", "Unknown")
            st.caption(f"File: {fname}")
            if "infer_schema" in by_tool:
                schema = by_tool["infer_schema"][0]["result"]
                shape  = schema.get("shape", {})
                c1, c2 = st.columns(2)
                c1.metric("Rows", shape.get("rows", "?"))
                c2.metric("Columns", shape.get("cols", "?"))
            analysis_plan = results.get("analysis_plan", "")
            if analysis_plan:
                with st.expander("Agent's Analysis Plan", expanded=True):
                    st.markdown(analysis_plan)

        # ----------------------------------------------------------------
        # ANALYSIS — schema, stats, visualizations, anomalies
        # ----------------------------------------------------------------
        elif selected == "analysis":
            st.markdown("**Analysis Results**")

            img_meta = results.get("image_processing_metadata")
            if img_meta:
                processed = img_meta.get("processed_count", 0)
                total     = img_meta.get("total_count", 0)
                coverage  = img_meta.get("coverage_percent", 0)
                st.info(f"Image dataset — {processed}/{total} images processed ({coverage:.1f}% coverage)")

            a1, a2, a3 = st.tabs(["Schema & Stats", "Visualizations", "Anomalies"])

            with a1:
                cl, cr = st.columns(2)
                with cl:
                    st.subheader("Schema")
                    if "infer_schema" in by_tool:
                        schema = by_tool["infer_schema"][0]["result"]
                        shape  = schema.get("shape", {})
                        st.metric("Rows", shape.get("rows", "?"))
                        st.metric("Columns", shape.get("cols", "?"))
                        roles  = schema.get("roles", {})
                        nulls  = schema.get("null_pct", {})
                        schema_df = pd.DataFrame({
                            "Column": list(roles.keys()),
                            "Role":   list(roles.values()),
                            "Null %": [nulls.get(c, 0) for c in roles],
                            "Dtype":  [schema.get("dtypes", {}).get(c, "") for c in roles],
                        })
                        st.dataframe(schema_df, use_container_width=True, hide_index=True)
                with cr:
                    st.subheader("Statistics")
                    if "summarize_statistics" in by_tool:
                        stats = by_tool["summarize_statistics"][0]["result"].get("statistics", {})
                        if stats:
                            stats_df = pd.DataFrame(stats).T
                            st.dataframe(
                                stats_df.style.format("{:.3f}", na_rep="—"),
                                use_container_width=True,
                            )
                if "compute_correlations" in by_tool:
                    st.subheader("Top Correlations")
                    top = by_tool["compute_correlations"][0]["result"].get("top_correlations", [])
                    if top:
                        st.dataframe(pd.DataFrame(top), use_container_width=True, hide_index=True)

            with a2:
                if not plot_paths:
                    st.info("No plots were generated.")
                else:
                    fname = st.session_state.get("uploaded_filename", "analysis")
                    st.download_button(
                        f"Download all {len(plot_paths)} plots (.zip)",
                        data=_build_plots_zip(plot_paths),
                        file_name=f"{os.path.splitext(fname)[0]}_plots.zip",
                        mime="application/zip",
                    )
                    st.divider()
                    cols = st.columns(2)
                    for i, path in enumerate(plot_paths):
                        if os.path.exists(path):
                            with cols[i % 2]:
                                st.image(Image.open(path), caption=os.path.basename(path),
                                         use_container_width=True)

            with a3:
                anomaly_entries = by_tool.get("detect_anomalies", [])
                if not anomaly_entries:
                    st.info("No anomaly detection was run.")
                else:
                    for entry in anomaly_entries:
                        r   = entry["result"]
                        col = r.get("column", entry["args"].get("column", "?"))
                        cnt = r.get("outlier_count", 0)
                        pct = r.get("outlier_pct", 0)
                        with st.expander(f"{col} — {cnt} outliers ({pct}%)"):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Count", cnt)
                            c2.metric("Pct", f"{pct}%")
                            bounds = r.get("iqr_bounds", {})
                            c3.metric("IQR Bounds",
                                      f"[{bounds.get('lower','?')}, {bounds.get('upper','?')}]")
                            indices = r.get("outlier_indices", [])
                            if indices:
                                st.caption(f"Outlier rows (first 50): {indices[:50]}")

        # ----------------------------------------------------------------
        # QUALITY — data quality audit
        # ----------------------------------------------------------------
        elif selected == "quality":
            st.markdown("**Detect Issues**")
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
                grade     = ("Excellent" if overall >= 90 else "Good" if overall >= 75
                             else "Fair" if overall >= 55 else "Poor")

                l, r_col = st.columns([1, 2], gap="large")
                with l:
                    st.metric("Overall Score", f"{overall} / 100", delta=grade, delta_color="off")
                    st.progress(int(overall) / 100)
                with r_col:
                    for lbl, val, tip in [
                        ("Completeness", breakdown.get("completeness", 0), "Non-null values"),
                        ("Uniqueness",   breakdown.get("uniqueness",   0), "Non-duplicate rows"),
                        ("Consistency",  breakdown.get("consistency",  0), "No mixed types"),
                    ]:
                        st.caption(f"**{lbl}** — {val:.1f}%  _{tip}_")
                        st.progress(int(val) / 100)

                st.divider()
                st.subheader("Column Health")
                schema_result = by_tool.get("infer_schema", [{}])[0].get("result", {})
                roles     = schema_result.get("roles", {})
                null_pcts = schema_result.get("null_pct", {})
                issue_cols = {i["column"] for i in issues}
                rows = []
                for col, role in roles.items():
                    null_p = null_pcts.get(col, 0)
                    if null_p == 0 and col not in issue_cols:
                        status = "OK"
                    elif null_p > 50 or any(
                        i["severity"] == "high" for i in issues if i["column"] == col
                    ):
                        status = "High"
                    else:
                        status = "Medium"
                    rows.append({"Column": col, "Type": role,
                                 "Complete": f"{100-null_p:.1f}%",
                                 "Missing": f"{null_p:.1f}%", "Status": status})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.divider()
                dup_result    = q_by_tool.get("detect_duplicates", [{}])[0].get("result", {})
                dup_count     = dup_result.get("duplicate_rows", 0)
                high_issues   = [i for i in issues if i["severity"] == "high"]
                medium_issues = [i for i in issues if i["severity"] == "medium"]

                if "dismissed_issues" not in st.session_state:
                    st.session_state["dismissed_issues"] = set()

                total_issues = len(issues) + (1 if dup_count else 0)
                if total_issues == 0:
                    st.success("No issues detected — this dataset looks clean.")
                else:
                    dismissed    = st.session_state["dismissed_issues"]
                    active_count = sum(
                        1 for i in high_issues + medium_issues
                        if f"{i['column']}|{i['issue']}" not in dismissed
                    ) + (1 if dup_count and "duplicates|duplicate_rows" not in dismissed else 0)

                    st.subheader(f"Issues  ({active_count} active)")
                    st.caption("Dismiss issues already handled — the Solutions node will skip them.")

                    for issue in high_issues + medium_issues:
                        issue_key   = f"{issue['column']}|{issue['issue']}"
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
                            lbl = "Restore" if is_dismissed else "Dismiss"
                            if st.button(lbl, key=f"dismiss_{issue_key}", use_container_width=True):
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
                                    f"({dup_result.get('duplicate_pct', 0):.1f}%)~~ _(dismissed)_"
                                )
                            else:
                                st.warning(
                                    f"**Duplicate rows** — {dup_count} rows "
                                    f"({dup_result.get('duplicate_pct', 0):.1f}%) are exact duplicates."
                                )
                        with col_btn:
                            lbl = "Restore" if is_dup_dismissed else "Dismiss"
                            if st.button(lbl, key=f"dismiss_{dup_key}", use_container_width=True):
                                if is_dup_dismissed:
                                    st.session_state["dismissed_issues"].discard(dup_key)
                                else:
                                    st.session_state["dismissed_issues"].add(dup_key)
                                st.rerun()
                        if not is_dup_dismissed:
                            examples = dup_result.get("examples", [])
                            if examples:
                                with st.expander("Show examples"):
                                    st.dataframe(pd.DataFrame(examples),
                                                 use_container_width=True, hide_index=True)

                if q_plot_paths:
                    st.divider()
                    heatmap_paths = [p for p in q_plot_paths if "missing" in os.path.basename(p)]
                    qq_paths      = [p for p in q_plot_paths if "qq_" in os.path.basename(p)]
                    if heatmap_paths:
                        st.subheader("Missing Value Map")
                        for path in heatmap_paths:
                            if os.path.exists(path):
                                st.image(Image.open(path), use_container_width=True)
                    if qq_paths:
                        st.subheader("Q-Q Plots")
                        st.caption("Points close to the line indicate a normal distribution.")
                        cols = st.columns(3)
                        for i, path in enumerate(qq_paths):
                            if os.path.exists(path):
                                with cols[i % 3]:
                                    st.image(Image.open(path),
                                             caption=os.path.basename(path).replace("qq_", "").replace(".png", ""),
                                             use_container_width=True)

                if q_narrative:
                    st.divider()
                    st.subheader("Quality Assessment")
                    st.markdown(q_narrative)

        # ----------------------------------------------------------------
        # SOLUTIONS — suggest fixes + apply
        # ----------------------------------------------------------------
        elif selected == "solutions":
            st.markdown("**Suggest Fixes**")
            solutions_narrative    = results.get("solutions_narrative", "")
            solutions_tool_results = results.get("solutions_tool_results", [])

            if not solutions_narrative and not solutions_tool_results:
                st.info("No data quality issues detected — no remediation needed.")
            else:
                if "selected_solutions" not in st.session_state:
                    st.session_state["selected_solutions"] = set()
                if "applied_solutions" not in st.session_state:
                    st.session_state["applied_solutions"] = []

                dismissed_issues = st.session_state.get("dismissed_issues", set())

                # Applied fixes banner
                applied     = st.session_state["applied_solutions"]
                _orig_bytes = st.session_state.get("original_csv_bytes")
                _csv_path   = st.session_state.get("csv_path", "")
                if not _orig_bytes and _csv_path and os.path.exists(_csv_path):
                    try:
                        with open(_csv_path, "rb") as _fh:
                            _orig_bytes = _fh.read()
                    except OSError:
                        pass

                if applied and _orig_bytes:
                    st.markdown(f"**{len(applied)} fix(es) applied**")
                    try:
                        _df = pd.read_csv(io.BytesIO(_orig_bytes))
                        for item in applied:
                            try:
                                exec(item["code"], {"df": _df, "pd": pd})  # noqa: S102
                            except Exception as _exc:
                                st.warning(f"Could not apply `{item['label']}`: {_exc}")
                        _cleaned_csv = _df.to_csv(index=False).encode()
                        _orig_fname  = st.session_state.get("uploaded_filename", "data")
                        st.download_button(
                            "Download cleaned CSV",
                            data=_cleaned_csv,
                            file_name=f"{os.path.splitext(_orig_fname)[0]}_cleaned.csv",
                            mime="text/csv",
                        )
                    except Exception as _e:
                        st.warning(f"Could not build cleaned CSV: {_e}")
                    if st.button("Clear all applied fixes"):
                        st.session_state["applied_solutions"] = []
                        st.rerun()
                    st.divider()

                if solutions_narrative:
                    with st.expander("Remediation Strategy", expanded=True):
                        st.markdown(solutions_narrative)
                    st.divider()

                if solutions_tool_results:
                    for tool_result in solutions_tool_results:
                        if tool_result["tool"] != "recommend_solutions":
                            continue
                        recommendations = tool_result["result"].get("recommendations", [])
                        if not recommendations:
                            continue

                        st.subheader("Actionable Solutions by Issue")
                        st.caption("Check actions for the report. Click Apply Fix to run the transformation.")

                        _widget_idx = 0
                        for rec in recommendations:
                            col_name   = rec.get("column", "—")
                            issue_type = rec.get("issue_type", "unknown")
                            severity   = rec.get("severity", "medium")
                            actions    = rec.get("actions", [])

                            if any(col_name in dk for dk in dismissed_issues):
                                continue

                            severity_text = {"high": "High", "medium": "Medium", "low": "Low"}.get(
                                severity, severity
                            )

                            with st.expander(f"{col_name} — {issue_type}"):
                                st.caption(f"Severity: **{severity_text}**")
                                for action in actions:
                                    priority       = action.get("priority", "medium").upper()
                                    action_name    = action.get("action", "")
                                    rationale      = action.get("rationale", "")
                                    implementation = action.get("implementation", "")

                                    action_key = f"{col_name}|{issue_type}|{action_name}"
                                    is_checked = action_key in st.session_state["selected_solutions"]
                                    is_applied = any(
                                        a["key"] == action_key
                                        for a in st.session_state["applied_solutions"]
                                    )

                                    check_col, apply_col = st.columns([5, 1])
                                    with check_col:
                                        checked = st.checkbox(
                                            f"**[{priority}]** {action_name}"
                                            + (" — applied" if is_applied else ""),
                                            value=is_checked,
                                            key=f"sol_{_widget_idx}",
                                        )
                                    _widget_idx += 1
                                    if checked:
                                        st.session_state["selected_solutions"].add(action_key)
                                    else:
                                        st.session_state["selected_solutions"].discard(action_key)

                                    with apply_col:
                                        if implementation:
                                            btn_lbl = "Undo" if is_applied else "Apply Fix"
                                            if st.button(btn_lbl, key=f"apply_{_widget_idx}",
                                                         use_container_width=True):
                                                if is_applied:
                                                    st.session_state["applied_solutions"] = [
                                                        a for a in st.session_state["applied_solutions"]
                                                        if a["key"] != action_key
                                                    ]
                                                else:
                                                    st.session_state["applied_solutions"].append({
                                                        "key":   action_key,
                                                        "label": action_name,
                                                        "code":  implementation,
                                                    })
                                                st.rerun()

                                    st.caption(f"**Rationale:** {rationale}")
                                    if implementation:
                                        st.code(implementation, language="python")
                                    st.divider()

                sel_count = len(st.session_state["selected_solutions"])
                if sel_count:
                    st.info(f"{sel_count} action(s) selected — the exported report will include only these.")

        # ----------------------------------------------------------------
        # REPORT — narrative + downloads
        # ----------------------------------------------------------------
        elif selected == "report":
            st.markdown("**Generated Report**")
            fname = st.session_state.get("uploaded_filename", "analysis")
            _sel  = st.session_state.get("selected_solutions") or None
            md_report = _build_markdown_report(results, fname, selected_solutions=_sel)

            if _sel:
                st.info(f"Report filtered to {len(_sel)} selected solution(s).")

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "Download Report (.md)",
                    data=md_report,
                    file_name=f"{os.path.splitext(fname)[0]}_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            if results.get("plot_paths"):
                with dl2:
                    st.download_button(
                        "Download Plots (.zip)",
                        data=_build_plots_zip(results["plot_paths"]),
                        file_name=f"{os.path.splitext(fname)[0]}_plots.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )

            st.divider()
            st.markdown(narrative)
