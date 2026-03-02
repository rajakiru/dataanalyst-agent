"""
MCP Server: Data Analysis Tools
Exposes 6 tools via stdio transport for the collection agent to call.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Server("data-analysis-server")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="infer_schema",
            description=(
                "Read a CSV file and return its schema: column names, inferred data types, "
                "shape (rows x cols), null counts per column, and column roles "
                "(numeric, categorical, datetime, or id)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."}
                },
                "required": ["csv_path"],
            },
        ),
        Tool(
            name="summarize_statistics",
            description=(
                "Compute descriptive statistics for all numeric columns: mean, std, min, max, "
                "quartiles, skewness, and kurtosis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."}
                },
                "required": ["csv_path"],
            },
        ),
        Tool(
            name="compute_correlations",
            description=(
                "Compute the Pearson correlation matrix for all numeric columns. "
                "Returns the matrix as a nested dict and lists the top 5 strongest correlations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."}
                },
                "required": ["csv_path"],
            },
        ),
        Tool(
            name="detect_anomalies",
            description=(
                "Detect outliers in a specific numeric column using the IQR method. "
                "Returns the count of outliers, their indices, and the IQR bounds used."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                    "column": {"type": "string", "description": "Name of the numeric column to analyse."},
                },
                "required": ["csv_path", "column"],
            },
        ),
        Tool(
            name="plot_distribution",
            description=(
                "Generate a histogram with a KDE overlay for a specific column and save it as a PNG. "
                "Returns the path to the saved image."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                    "column": {"type": "string", "description": "Name of the column to plot."},
                },
                "required": ["csv_path", "column"],
            },
        ),
        Tool(
            name="plot_correlation_heatmap",
            description=(
                "Generate a Seaborn correlation heatmap for all numeric columns and save it as a PNG. "
                "Returns the path to the saved image."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."}
                },
                "required": ["csv_path"],
            },
        ),
    ]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "infer_schema":
        result = _infer_schema(arguments["csv_path"])
    elif name == "summarize_statistics":
        result = _summarize_statistics(arguments["csv_path"])
    elif name == "compute_correlations":
        result = _compute_correlations(arguments["csv_path"])
    elif name == "detect_anomalies":
        result = _detect_anomalies(arguments["csv_path"], arguments["column"])
    elif name == "plot_distribution":
        result = _plot_distribution(arguments["csv_path"], arguments["column"])
    elif name == "plot_correlation_heatmap":
        result = _plot_correlation_heatmap(arguments["csv_path"])
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------

def _load(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _infer_schema(csv_path: str) -> dict:
    df = _load(csv_path)
    roles = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            roles[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            # heuristic: if int with low cardinality, treat as categorical
            if df[col].dtype in ["int64", "int32"] and df[col].nunique() <= 10:
                roles[col] = "categorical"
            else:
                roles[col] = "numeric"
        elif df[col].nunique() == len(df):
            roles[col] = "id"
        else:
            roles[col] = "categorical"

    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": {col: int(n) for col, n in df.isnull().sum().items()},
        "null_pct": {col: round(float(n) / len(df) * 100, 2)
                     for col, n in df.isnull().sum().items()},
        "roles": roles,
        "sample_values": {col: df[col].dropna().head(3).tolist() for col in df.columns},
    }


def _summarize_statistics(csv_path: str) -> dict:
    df = _load(csv_path)
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return {"error": "No numeric columns found."}

    desc = numeric.describe().round(4).to_dict()
    skew = numeric.skew().round(4).to_dict()
    kurt = numeric.kurtosis().round(4).to_dict()

    # merge into one dict per column
    stats = {}
    for col in numeric.columns:
        stats[col] = desc.get(col, {})
        stats[col]["skewness"] = skew.get(col)
        stats[col]["kurtosis"] = kurt.get(col)

    return {"statistics": stats}


def _compute_correlations(csv_path: str) -> dict:
    df = _load(csv_path)
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for correlation."}

    corr = numeric.corr().round(4)
    corr_dict = corr.to_dict()

    # top 5 strongest correlations (excluding self-correlations)
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                "col1": cols[i],
                "col2": cols[j],
                "correlation": round(float(corr.iloc[i, j]), 4),
            })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "correlation_matrix": corr_dict,
        "top_correlations": pairs[:5],
    }


def _detect_anomalies(csv_path: str, column: str) -> dict:
    df = _load(csv_path)
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric."}

    series = df[column].dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    outlier_mask = (df[column] < lower) | (df[column] > upper)
    outlier_indices = df.index[outlier_mask].tolist()

    return {
        "column": column,
        "iqr_bounds": {"lower": round(float(lower), 4), "upper": round(float(upper), 4)},
        "q1": round(float(q1), 4),
        "q3": round(float(q3), 4),
        "iqr": round(float(iqr), 4),
        "outlier_count": int(outlier_mask.sum()),
        "outlier_pct": round(float(outlier_mask.sum()) / len(df) * 100, 2),
        "outlier_indices": outlier_indices[:50],  # cap at 50
    }


def _plot_distribution(csv_path: str, column: str) -> dict:
    df = _load(csv_path)
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    fig, ax = plt.subplots(figsize=(7, 4))
    series = df[column].dropna()

    if pd.api.types.is_numeric_dtype(series):
        sns.histplot(series, kde=True, ax=ax, color="steelblue")
        ax.set_title(f"Distribution of {column}")
    else:
        value_counts = series.value_counts().head(15)
        value_counts.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(f"Value Counts: {column}")
        plt.xticks(rotation=45, ha="right")

    ax.set_xlabel(column)
    fig.tight_layout()

    safe_col = column.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"dist_{safe_col}.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return {"plot_path": out_path, "column": column}


def _plot_correlation_heatmap(csv_path: str) -> dict:
    df = _load(csv_path)
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for a heatmap."}

    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr.columns)), max(5, len(corr.columns) - 1)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return {"plot_path": out_path}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
