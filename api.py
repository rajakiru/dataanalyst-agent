"""
FastAPI backend — exposes /plan, /execute, /chat for the React frontend.
Run: uvicorn api:app --reload --port 8000
"""

import os
import tempfile
import pickle
import base64
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent import run_plan_only, run_execution_phase, DEFAULT_MODEL

app = FastAPI(title="AutoAnalyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"
SAMPLE_PATHS = {
    "iris":               DATA_DIR / "iris.csv",
    "iris_corrupted":     DATA_DIR / "iris_corrupted.csv",
    "titanic":            Path(__file__).parent / "Titanic-Dataset.csv",
    "titanic_corrupted":  DATA_DIR / "titanic_corrupted.csv",
}

API_KEY = os.getenv("DEDALUS_API_KEY", "")
MODEL   = os.getenv("DEFAULT_MODEL", DEFAULT_MODEL)


def _serialize_plan_state(plan_state: dict) -> dict:
    """Convert plan_state to a JSON-safe dict for the client."""
    return {
        "analysis_plan":        plan_state.get("analysis_plan", ""),
        "available_tool_names": plan_state.get("available_tool_names", []),
        "planned_tool_names":   plan_state.get("planned_tool_names", []),
        "tool_descriptions":    plan_state.get("tool_descriptions", {}),
        "enabled_categories":   list(plan_state.get("enabled_categories") or []),
        # Serialise the full state as base64-pickled blob so /execute can restore it
        "_blob": base64.b64encode(pickle.dumps(plan_state)).decode(),
    }


def _deserialize_plan_state(data: dict) -> dict:
    blob = data.get("_blob")
    if blob:
        return pickle.loads(base64.b64decode(blob))
    return data


def _make_results_json(results: dict) -> dict:
    """Strip non-JSON-serialisable objects; encode plots as base64 strings."""
    out = {}
    for k, v in results.items():
        if k == "plots":
            encoded = []
            for p in (v or []):
                if isinstance(p, bytes):
                    encoded.append(base64.b64encode(p).decode())
                elif isinstance(p, str):
                    encoded.append(p)
            out["plots"] = encoded
        elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
            out[k] = v
        else:
            try:
                import json
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# POST /plan  — phase 1: run collection agent planning only
# ---------------------------------------------------------------------------

@app.post("/plan")
async def plan(
    file: UploadFile | None = File(default=None),
    sample: str | None = Form(default=None),
):
    if file and file.filename:
        suffix = Path(file.filename).suffix or ".csv"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await file.read())
        tmp.flush()
        csv_path = tmp.name
        cleanup = True
    elif sample and sample in SAMPLE_PATHS:
        csv_path = str(SAMPLE_PATHS[sample])
        cleanup = False
    else:
        raise HTTPException(400, "Provide a file upload or a valid sample name")

    try:
        plan_state = run_plan_only(csv_path=csv_path, model=MODEL, api_key=API_KEY)
    finally:
        if cleanup:
            os.unlink(csv_path)

    return _serialize_plan_state(plan_state)


# ---------------------------------------------------------------------------
# POST /execute  — phase 2: run full pipeline with approved tools
# ---------------------------------------------------------------------------

class ExecuteRequest(BaseModel):
    plan_state: dict
    approved_tools: list[str]


@app.post("/execute")
async def execute(req: ExecuteRequest):
    plan_state = _deserialize_plan_state(req.plan_state)
    results = run_execution_phase(
        plan_state=plan_state,
        approved_tools=req.approved_tools,
        model=MODEL,
        api_key=API_KEY,
    )
    return _make_results_json(results)


# ---------------------------------------------------------------------------
# POST /chat  — simple Q&A against the LLM (no tool use)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    context: str = ""


@app.post("/chat")
async def chat(req: ChatRequest):
    from openai import OpenAI
    base_url = os.getenv("DEDALUS_BASE_URL", "https://api.dedaluslabs.ai/v1")
    client = OpenAI(api_key=API_KEY, base_url=base_url)

    system = (
        "You are a data analysis assistant for the AutoAnalyst system. "
        "Help the user interpret their dataset's quality issues, understand agent findings, "
        "and choose appropriate fixes. Be concise and practical."
    )
    if req.context:
        system += f"\n\nAnalysis context:\n{req.context}"

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": req.message},
        ],
        max_tokens=512,
    )
    return {"reply": resp.choices[0].message.content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
