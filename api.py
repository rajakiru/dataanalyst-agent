"""
FastAPI backend for AutoAnalyst React frontend.
Run: uvicorn api:app --reload --port 8000
"""

import asyncio
import io
import json
import os
import queue
import shutil
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

load_dotenv()

from agent import (
    DEFAULT_MODEL,
    _detect_and_process_images,
    run_execution_phase,
    run_plan_only,
)

app = FastAPI(title="AutoAnalyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
# session_id -> {csv_path, filename, plan_state, results, plot_dir}
_SESSIONS: dict[str, dict] = {}

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_FILES = {
    "Iris (clean)":        DATA_DIR / "iris.csv",
    "Iris (corrupted)":    DATA_DIR / "iris_corrupted.csv",
    "Titanic":             Path(__file__).parent / "Titanic-Dataset.csv",
    "Titanic (corrupted)": DATA_DIR / "titanic_corrupted.csv",
}

API_KEY = os.getenv("OPENAI_API_KEY", "")

PLOTS_TEMP_BASE = Path(tempfile.gettempdir()) / "autoanalyst_plots"
PLOTS_TEMP_BASE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_session(csv_path: str, filename: str) -> str:
    sid = str(uuid.uuid4())
    _SESSIONS[sid] = {
        "csv_path": csv_path,
        "filename": filename,
        "plan_state": None,
        "results": None,
        "plot_dir": None,
    }
    return sid


def _serialize_results(results: dict, session_id: str) -> dict:
    """Copy plot files to a temp dir and replace paths with API URLs."""
    plot_dir = PLOTS_TEMP_BASE / session_id
    plot_dir.mkdir(parents=True, exist_ok=True)
    _SESSIONS[session_id]["plot_dir"] = str(plot_dir)

    def _copy_plots(paths: list) -> list:
        urls = []
        for p in paths:
            if p and os.path.exists(p):
                fname = os.path.basename(p)
                dest = plot_dir / fname
                if not dest.exists():
                    shutil.copy2(p, dest)
                urls.append(f"/api/plots/{session_id}/{fname}")
        return urls

    out = dict(results)
    out["plot_paths"] = _copy_plots(results.get("plot_paths") or [])
    out["quality_plot_paths"] = _copy_plots(results.get("quality_plot_paths") or [])

    # Make sure tool_results is JSON-serialisable (drop non-serialisable values)
    def _clean(v):
        try:
            json.dumps(v)
            return v
        except (TypeError, ValueError):
            return str(v)

    out["tool_results"] = [
        {"tool": r["tool"], "args": r.get("args", {}), "result": _clean(r["result"])}
        for r in (results.get("tool_results") or [])
    ]
    out["quality_tool_results"] = [
        {"tool": r["tool"], "args": r.get("args", {}), "result": _clean(r["result"])}
        for r in (results.get("quality_tool_results") or [])
    ]
    out["solutions_tool_results"] = [
        {"tool": r["tool"], "args": r.get("args", {}), "result": _clean(r.get("result", {}))}
        for r in (results.get("solutions_tool_results") or [])
    ]
    return out


def _build_markdown_report(
    results: dict,
    filename: str,
    selected_solutions: Optional[set] = None,
) -> str:
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
        parts.append(f"- **Rows:** {shape.get('rows', '?')}  \n")
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
                parts.append(
                    f"| {issue['column']} | {issue['issue']} | {issue['severity']} |\n"
                )

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
            if tool_result["tool"] != "recommend_solutions":
                continue
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
                            a
                            for a in actions
                            if f"{col_name}|{issue_type}|{a.get('action', '')}"
                            in selected_solutions
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


# ---------------------------------------------------------------------------
# POST /api/upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.csv").suffix or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await file.read()
    tmp.write(content)
    tmp.flush()
    tmp.close()

    sid = _new_session(tmp.name, file.filename or "upload.csv")
    return {"session_id": sid, "filename": file.filename}


# ---------------------------------------------------------------------------
# POST /api/upload-sample
# ---------------------------------------------------------------------------

class UploadSampleRequest(BaseModel):
    name: str


@app.post("/api/upload-sample")
async def upload_sample(req: UploadSampleRequest):
    path = SAMPLE_FILES.get(req.name)
    if not path or not path.exists():
        raise HTTPException(400, f"Sample '{req.name}' not found")
    # Copy to temp so it behaves like an upload
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    shutil.copy2(str(path), tmp.name)
    tmp.close()
    sid = _new_session(tmp.name, path.name)
    return {"session_id": sid, "filename": path.name}


# ---------------------------------------------------------------------------
# GET /api/samples
# ---------------------------------------------------------------------------

@app.get("/api/samples")
async def list_samples():
    samples = []
    for name, path in SAMPLE_FILES.items():
        samples.append({"name": name, "available": path.exists()})
    return {"samples": samples}


# ---------------------------------------------------------------------------
# POST /api/plan
# ---------------------------------------------------------------------------

class PlanRequest(BaseModel):
    session_id: str
    model: str = DEFAULT_MODEL
    enabled_categories: list[str] = []


@app.post("/api/plan")
async def run_plan(req: PlanRequest):
    session = _SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    csv_path = session["csv_path"]
    enabled = frozenset(req.enabled_categories) if req.enabled_categories else None

    events: list[dict] = []

    def on_event(event: dict):
        events.append(event)

    try:
        plan_state = await asyncio.to_thread(
            run_plan_only,
            csv_path,
            req.model,
            API_KEY,
            enabled,
            on_event,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))

    session["plan_state"] = plan_state

    plan_data = {
        "analysis_plan": plan_state.get("analysis_plan", ""),
        "available_tool_names": plan_state.get("available_tool_names", []),
        "planned_tool_names": plan_state.get("planned_tool_names", []),
        "tool_descriptions": plan_state.get("tool_descriptions", {}),
        "enabled_categories": list(plan_state.get("enabled_categories") or []),
    }
    return {"plan": plan_data, "events": events}


# ---------------------------------------------------------------------------
# POST /api/execute/{session_id}  — SSE streaming
# ---------------------------------------------------------------------------

class ExecuteRequest(BaseModel):
    approved_tools: list[str]
    model: str = DEFAULT_MODEL


@app.post("/api/execute/{session_id}")
async def execute_stream(session_id: str, req: ExecuteRequest):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    plan_state = session.get("plan_state")
    if not plan_state:
        raise HTTPException(400, "No plan found for this session. Run /api/plan first.")

    event_queue: queue.Queue = queue.Queue()
    result_container: dict = {}

    def worker():
        def on_event(event: dict):
            event_queue.put(event)

        try:
            results = run_execution_phase(
                plan_state,
                req.approved_tools,
                req.model,
                API_KEY,
                on_event,
            )
            result_container["results"] = results
            event_queue.put({"type": "complete"})
        except Exception as exc:
            event_queue.put({"type": "error", "message": str(exc)})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()

    async def generate():
        while True:
            try:
                event = await loop.run_in_executor(None, event_queue.get)
            except Exception:
                break

            data = json.dumps(event)
            yield f"data: {data}\n\n"

            if event.get("type") in ("complete", "error"):
                break

        # Store results in session after completion
        if "results" in result_container:
            try:
                serialized = _serialize_results(result_container["results"], session_id)
                session["results"] = serialized
            except Exception as exc:
                session["results"] = {"error": str(exc)}

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/results/{session_id}
# ---------------------------------------------------------------------------

@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    results = session.get("results")
    if not results:
        raise HTTPException(404, "Results not ready yet")
    return results


# ---------------------------------------------------------------------------
# GET /api/plots/{session_id}/{filename}
# ---------------------------------------------------------------------------

@app.get("/api/plots/{session_id}/{filename}")
async def get_plot(session_id: str, filename: str):
    plot_dir = PLOTS_TEMP_BASE / session_id
    file_path = plot_dir / filename
    if not file_path.exists():
        raise HTTPException(404, "Plot not found")
    return FileResponse(str(file_path), media_type="image/png")


# ---------------------------------------------------------------------------
# POST /api/download/report/{session_id}
# ---------------------------------------------------------------------------

class ReportRequest(BaseModel):
    selected_solutions: list[str] = []


@app.post("/api/download/report/{session_id}")
async def download_report(session_id: str, req: ReportRequest):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    results = session.get("results")
    if not results:
        raise HTTPException(404, "Results not ready")

    sel = set(req.selected_solutions) if req.selected_solutions else None
    filename = session.get("filename", "analysis")
    md = _build_markdown_report(results, filename, sel)
    stem = Path(filename).stem

    return Response(
        content=md.encode("utf-8"),
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{stem}_report.md"'},
    )


# ---------------------------------------------------------------------------
# POST /api/download/cleaned-csv/{session_id}
# ---------------------------------------------------------------------------

class CleanedCsvRequest(BaseModel):
    applied_solutions: list[dict] = []  # [{key, label, code}]


@app.post("/api/download/cleaned-csv/{session_id}")
async def download_cleaned_csv(session_id: str, req: CleanedCsvRequest):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    csv_path = session.get("csv_path", "")
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(400, "Original CSV not available")

    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        for item in req.applied_solutions:
            code = item.get("code", "")
            if code:
                try:
                    exec(code, {"df": df, "pd": pd})  # noqa: S102
                except Exception:
                    pass

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv_bytes = buf.getvalue().encode("utf-8")
    except Exception as exc:
        raise HTTPException(500, f"Failed to build cleaned CSV: {exc}")

    filename = session.get("filename", "data.csv")
    stem = Path(filename).stem

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{stem}_cleaned.csv"'},
    )


# ---------------------------------------------------------------------------
# POST /api/simulate-fix/{session_id}  — preview score change after applying fixes
# ---------------------------------------------------------------------------

def _compute_quality_from_df(df) -> dict:
    """Replicates mcp_server._compute_data_quality_score logic on a DataFrame."""
    import numpy as np

    total = len(df)
    if total == 0:
        return {"overall_score": 0, "issue_count": 0}

    # Completeness
    non_null = df.notnull().sum().sum()
    completeness = (non_null / (total * len(df.columns))) * 100 if len(df.columns) else 100

    # Uniqueness
    dup_rows = df.duplicated().sum()
    uniqueness = ((total - dup_rows) / total) * 100

    # Consistency: penalise zero-variance numeric cols and mixed-type cols
    penalty = 0
    for col in df.columns:
        if df[col].dtype in (float, int) or np.issubdtype(df[col].dtype, np.number):
            if df[col].nunique() <= 1:
                penalty += 10
        else:
            non_null_vals = df[col].dropna()
            if len(non_null_vals) > 0:
                types = non_null_vals.apply(type).nunique()
                if types > 1:
                    penalty += 5
    consistency = max(0.0, 100.0 - penalty)

    overall = completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3

    # Count issues (simple heuristic matching mcp_server logic)
    issues = []
    for col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        if null_pct > 5:
            sev = "high" if null_pct > 20 else "medium"
            issues.append({"column": col, "severity": sev})
    if dup_rows > 0:
        issues.append({"column": "all", "severity": "high" if dup_rows / total > 0.05 else "medium"})

    return {"overall_score": round(overall), "issue_count": len(issues)}


class SimulateFixRequest(BaseModel):
    applied_solutions: list[dict] = []  # [{key, label, code}]


@app.post("/api/simulate-fix/{session_id}")
async def simulate_fix(session_id: str, req: SimulateFixRequest):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    csv_path = session.get("csv_path", "")
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(400, "Original CSV not available")

    try:
        import pandas as pd

        df_before = pd.read_csv(csv_path)
        before = _compute_quality_from_df(df_before)

        df_after = df_before.copy()
        for item in req.applied_solutions:
            code = item.get("code", "")
            if code:
                try:
                    exec(code, {"df": df_after, "pd": pd})  # noqa: S102
                except Exception:
                    pass

        after = _compute_quality_from_df(df_after)

    except Exception as exc:
        raise HTTPException(500, f"Simulation failed: {exc}")

    return {
        "before": before,
        "after": after,
        "delta": after["overall_score"] - before["overall_score"],
    }


# ---------------------------------------------------------------------------
# GET /api/plots/zip/{session_id}  — download all plots as ZIP
# ---------------------------------------------------------------------------

@app.get("/api/plots/zip/{session_id}")
async def download_plots_zip(session_id: str):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    results = session.get("results", {})
    all_plot_urls = (results.get("plot_paths") or []) + (
        results.get("quality_plot_paths") or []
    )

    plot_dir = PLOTS_TEMP_BASE / session_id
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for url in all_plot_urls:
            fname = url.split("/")[-1]
            fpath = plot_dir / fname
            if fpath.exists():
                zf.write(str(fpath), fname)

    buf.seek(0)
    filename = session.get("filename", "analysis")
    stem = Path(filename).stem

    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{stem}_plots.zip"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
