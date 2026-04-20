# AutoAnalyst: A Multi-Agent Data Cascade Debugger

AutoAnalyst is an agent-driven system that automatically analyzes datasets, identifies data quality issues, explains their root causes, and proposes actionable fixes with measurable impact. The system uses a **Model Context Protocol (MCP)** architecture with multiple specialized agents (collection, quality, solutions, reporting) and a React workflow-builder UI with a FastAPI backend.

Unlike traditional AutoEDA tools that only surface insights, AutoAnalyst closes the loop by enabling users to understand, fix, and validate data issues — directly addressing **data cascades**, where unresolved upstream data problems silently degrade downstream AI systems.

---

## How to Run (current)

### 1. Backend (FastAPI)
```bash
cd dataanalyst-agent
source venv/bin/activate
uvicorn api:app --port 8000 --reload
```

### 2. Frontend (React + Vite)
```bash
cd react-frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

> The Vite dev server proxies all `/api` requests to `localhost:8000` automatically.

### 3. (Optional) Streamlit app — legacy
```bash
streamlit run app.py
```
Open `http://localhost:8501`. The Streamlit app is feature-complete but the React UI is the primary interface.

---

## Setup (first time)

**Requirements:** Python 3.11+, Node 18+

```bash
# 1. Clone
git clone https://github.com/rajakiru/dataanalyst-agent.git
cd dataanalyst-agent

# 2. Python venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Environment
cp .env.example .env
# Fill in OPENAI_API_KEY (and optionally DEDALUS_API_KEY / DEDALUS_BASE_URL)

# 4. Frontend dependencies
cd react-frontend
npm install
```

**.env:**
```
OPENAI_API_KEY=sk-...
DEDALUS_API_KEY=your_key          # optional Dedalus proxy
DEDALUS_BASE_URL=https://api.dedaluslabs.ai/v1
DEFAULT_MODEL=gpt-4o-mini
```

---

## What's Done

### Core pipeline
- **4-agent MCP architecture** — Collection → Quality → Solutions → Reporting agents
- **15+ MCP tools** — schema inference, statistics, correlations, anomaly detection, quality scoring, duplicate detection, solution recommendation
- **Human-in-the-loop** — agent presents analysis plan, user reviews/toggles tools before execution
- **Live agent timeline** — SSE streaming shows each phase and tool call in real time as they happen
- **Image dataset support** — ResNet18 or color-histogram feature extraction, processes image directories as CSV

### React UI (`react-frontend/`)
- **Workflow canvas** — pipeline nodes (Dataset → Run Analysis → Detect Issues → Suggest Fixes → Generate Report) with live status badges
- **Editor / Dashboard mode toggle** — Editor routes right panel by selected node; Dashboard shows aggregate health overview
- **Resizable split** — drag the divider between canvas and panel
- **Quality panel** — health score, breakdown (completeness/uniqueness/consistency), column health table, issue cards
- **Solutions panel** — per-issue fix recommendations with rationale, priority, and Python implementation code
- **Fix Preview modal** — before/after data diff showing which rows and cells change when a fix is applied (red = before, green = after)
- **Simulate Fix** — applies selected fixes and re-computes quality score to show delta before committing
- **Report panel** — full narrative export as markdown
- **Analysis panel** — plots (correlation heatmap, distributions, Q-Q plots), statistics table, anomaly results

### Sample datasets
| Sample | Description | Load time |
|---|---|---|
| Iris (clean) | 150 rows, 5 cols — baseline clean dataset | ~45s (live) |
| Iris (corrupted) | Iris with injected missing values and type errors | ~45s (live) |
| Titanic | 891 rows, 12 cols — survival data | ~60s (live) |
| **Titanic (corrupted)** | Corrupted Titanic — recommended for demo | **instant** (cached) |

The **Titanic (corrupted)** sample is pre-computed and loads in ~1 second while still playing through the full human-in-the-loop review and animated agent timeline.

### API endpoints (`api.py`)
| Endpoint | Description |
|---|---|
| `POST /api/upload` | Upload a CSV file |
| `POST /api/upload-sample` | Load a built-in sample dataset |
| `POST /api/load-cache` | Instant-load a pre-computed cached sample |
| `POST /api/plan` | Run planning phase, returns tool plan for review |
| `POST /api/execute/{id}` | Run execution phase via SSE streaming |
| `GET /api/results/{id}` | Fetch full results after execution |
| `POST /api/simulate-fix/{id}` | Preview quality score change after applying fixes |
| `POST /api/preview-fix/{id}` | Get before/after sample rows for a fix |
| `POST /api/download/report/{id}` | Download markdown report |
| `POST /api/download/cleaned-csv/{id}` | Download CSV with fixes applied |

---

## In Progress

### Image data cascade demo
Demonstrating a full cascade scenario with image data:
- **"Flowers (corrupted)" sample** — 20 synthetic flower images with 4 deliberately deleted to simulate missing training data
- Pipeline detects 80% image coverage → quality score drops → cascade risk flagged
- **Solutions agent recommends DALL-E image generation** as the fix
- **Preview fix** → DALL-E 2 generates a synthetic replacement flower image, shown live in modal
- **Story**: missing image → failed feature extraction → NaN rows → model bias → DALL-E fix → cascade resolved

To generate the flower dataset manually:
```bash
venv/bin/python -c "
from generate_sample_images import generate_flower_images
generate_flower_images(output_dir='data/flowers', num_images=20)
"
```

---

## Architecture

```
react-frontend/     (Vite + React + Tailwind, port 3000)
    └── api.py      (FastAPI backend, port 8000)
            └── agent.py
                    ├── Collection Agent   ←→  OpenAI API
                    │       └── MCP Client (stdio) → mcp_server.py
                    ├── Quality Agent      ←→  OpenAI API
                    │       └── MCP Client → quality + duplicate tools
                    ├── Solutions Agent    ←→  OpenAI API
                    │       └── MCP Client → recommend_solutions tool
                    └── Reporting Agent    ←→  OpenAI API
```

MCP tools in `mcp_server.py`:
`infer_schema` · `summarize_statistics` · `compute_correlations` · `detect_anomalies` · `plot_distribution` · `plot_correlation_heatmap` · `plot_pairplot` · `compute_data_quality_score` · `detect_duplicates` · `plot_missing_heatmap` · `plot_qq` · `recommend_solutions` · `process_images_to_csv`

---

## Project

Built for CMU 24-780 AI Agents (Spring 2026) by Kiruthika Raja & Akshara.
