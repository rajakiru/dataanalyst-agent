# AutoAnalyst: Multi-Agent Data Analysis System

An automated data analysis pipeline using the **Model Context Protocol (MCP)**. Upload any CSV file and a multi-agent system autonomously runs statistical analysis, generates visualizations, detects anomalies, and produces a written report — no manual configuration needed.

---

## How It Works

1. **Collection Agent** — reads the CSV, infers the schema, and decides which analysis tools to call
2. **MCP Server** — exposes 6 analysis tools the agent can invoke via stdio transport
3. **Reporting Agent** — takes all tool outputs and writes a narrative summary
4. **Streamlit Dashboard** — displays everything in a tabbed UI

---

## Setup

**Requirements:** Python 3.11+

```bash
# 1. Clone the repo
git clone https://github.com/rajakiru/dataanalyst-agent.git
cd dataanalyst-agent

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your Dedalus API key
```

**.env file:**
```
DEDALUS_API_KEY=your_key_here
DEDALUS_BASE_URL=https://api.dedaluslabs.ai/v1
DEFAULT_MODEL=openai/gpt-4o-mini
```

---

## Running

**Streamlit app (recommended):**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser, upload a CSV, and click **Run Analysis**.

**Command line (for testing):**
```bash
python agent.py data/iris.csv
```

---

## What It Does

The agent autonomously performs the following for any uploaded CSV:

| Analysis | Description |
|---|---|
| **Schema inference** | Detects column types, null counts, and column roles (numeric, categorical, datetime, ID) |
| **Descriptive statistics** | Mean, std, min, max, quartiles, skewness, kurtosis for all numeric columns |
| **Correlation analysis** | Pearson correlation matrix and top correlated column pairs |
| **Distribution plots** | Histogram + KDE overlay for each numeric/categorical column |
| **Correlation heatmap** | Seaborn heatmap of the full correlation matrix |
| **Anomaly detection** | IQR-based outlier detection with bounds and flagged row indices |
| **Narrative report** | 3–5 paragraph plain-English summary of findings |

Results are displayed across 4 tabs:
- **Schema & Stats** — data types, null percentages, descriptive statistics table, top correlations
- **Visualizations** — all generated plots in a 2-column grid
- **Anomalies** — per-column outlier counts, IQR bounds, flagged row indices
- **Report** — full written narrative from the reporting agent

---

## Test Datasets

The repo includes `data/iris.csv` (150 rows, UCI Iris dataset) to get started.

For a more complex test, try the **Titanic dataset**:
[Download from Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset?resource=download)

---

## Architecture

```
app.py (Streamlit UI)
    └── agent.py
            ├── Collection Agent  ←→  Dedalus LLM API (OpenAI-compatible)
            │       └── MCP Client (stdio)
            │               └── mcp_server.py (6 tools)
            └── Reporting Agent  ←→  Dedalus LLM API
```

Tools in `mcp_server.py`: `infer_schema`, `summarize_statistics`, `compute_correlations`, `detect_anomalies`, `plot_distribution`, `plot_correlation_heatmap`

---

## Project

Built for CMU 24-880 AI Agents (Spring 2026).
