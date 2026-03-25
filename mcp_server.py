"""
MCP Server: Data Analysis Tools
Exposes 14 tools via stdio transport for the collection agent to call.
"""

import json
import os
import sys
from scipy import stats as scipy_stats

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
        Tool(
            name="plot_scatter",
            description=(
                "Generate a scatter plot with a regression line for two numeric columns. "
                "Use this for pairs of columns with strong correlations. "
                "Returns the path to the saved PNG."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                    "x_column": {"type": "string", "description": "Name of the column for the x-axis."},
                    "y_column": {"type": "string", "description": "Name of the column for the y-axis."},
                },
                "required": ["csv_path", "x_column", "y_column"],
            },
        ),
        Tool(
            name="plot_boxplot",
            description=(
                "Generate a box plot showing the distribution of a numeric column grouped by a "
                "categorical column. Use this to compare groups (e.g., species vs petal_length). "
                "Returns the path to the saved PNG."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                    "numeric_column": {"type": "string", "description": "Numeric column to plot on the y-axis."},
                    "category_column": {"type": "string", "description": "Categorical column to group by on the x-axis."},
                },
                "required": ["csv_path", "numeric_column", "category_column"],
            },
        ),
        Tool(
            name="test_normality",
            description=(
                "Run a normality test on a numeric column. Uses Shapiro-Wilk for datasets under "
                "5000 rows, otherwise D'Agostino-Pearson. Returns the test statistic, p-value, "
                "and whether the column is approximately normal."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                    "column": {"type": "string", "description": "Name of the numeric column to test."},
                },
                "required": ["csv_path", "column"],
            },
        ),
        Tool(
            name="plot_pairplot",
            description=(
                "Generate a Seaborn pairplot showing pairwise relationships across all numeric columns. "
                "If a categorical column is present, it will be used as the hue. "
                "Capped at 6 numeric columns for readability. Returns the path to the saved PNG."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                },
                "required": ["csv_path"],
            },
        ),
        Tool(
            name="compute_data_quality_score",
            description=(
                "Compute an overall data quality score (0-100) and a breakdown across three dimensions: "
                "completeness (non-null ratio), uniqueness (non-duplicate ratio), and consistency "
                "(type uniformity and low-variance detection). Also returns a list of specific issues found."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                },
                "required": ["csv_path"],
            },
        ),
        Tool(
            name="detect_duplicates",
            description=(
                "Detect duplicate rows in the dataset. Returns the count, percentage, and the "
                "first 5 duplicate rows as examples."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                },
                "required": ["csv_path"],
            },
        ),
        Tool(
            name="plot_missing_heatmap",
            description=(
                "Generate a heatmap showing which cells in the dataset are missing (null). "
                "Only generates a plot if missing values exist. Returns the path to the saved PNG."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                },
                "required": ["csv_path"],
            },
        ),
        Tool(
            name="plot_qq",
            description=(
                "Generate a Q-Q (quantile-quantile) plot for a numeric column to visually assess "
                "whether it follows a normal distribution. Returns the path to the saved PNG."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                    "column": {"type": "string", "description": "Name of the numeric column to plot."},
                },
                "required": ["csv_path", "column"],
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
    elif name == "plot_scatter":
        result = _plot_scatter(arguments["csv_path"], arguments["x_column"], arguments["y_column"])
    elif name == "plot_boxplot":
        result = _plot_boxplot(arguments["csv_path"], arguments["numeric_column"], arguments["category_column"])
    elif name == "test_normality":
        result = _test_normality(arguments["csv_path"], arguments["column"])
    elif name == "plot_pairplot":
        result = _plot_pairplot(arguments["csv_path"])
    elif name == "compute_data_quality_score":
        result = _compute_data_quality_score(arguments["csv_path"])
    elif name == "detect_duplicates":
        result = _detect_duplicates(arguments["csv_path"])
    elif name == "plot_missing_heatmap":
        result = _plot_missing_heatmap(arguments["csv_path"])
    elif name == "plot_qq":
        result = _plot_qq(arguments["csv_path"], arguments["column"])
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


def _plot_scatter(csv_path: str, x_column: str, y_column: str) -> dict:
    df = _load(csv_path)
    for col in [x_column, y_column]:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found."}
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"error": f"Column '{col}' is not numeric."}

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(data=df, x=x_column, y=y_column, ax=ax,
                scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
    ax.set_title(f"{x_column} vs {y_column}")
    fig.tight_layout()

    safe_x = x_column.replace("/", "_").replace(" ", "_")
    safe_y = y_column.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"scatter_{safe_x}_vs_{safe_y}.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return {"plot_path": out_path, "x_column": x_column, "y_column": y_column}


def _plot_boxplot(csv_path: str, numeric_column: str, category_column: str) -> dict:
    df = _load(csv_path)
    if numeric_column not in df.columns:
        return {"error": f"Column '{numeric_column}' not found."}
    if category_column not in df.columns:
        return {"error": f"Column '{category_column}' not found."}
    if not pd.api.types.is_numeric_dtype(df[numeric_column]):
        return {"error": f"Column '{numeric_column}' is not numeric."}

    n_categories = df[category_column].nunique()
    fig_width = max(6, n_categories * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    sns.boxplot(data=df, x=category_column, y=numeric_column, ax=ax, palette="Set2")
    ax.set_title(f"{numeric_column} by {category_column}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    safe_num = numeric_column.replace("/", "_").replace(" ", "_")
    safe_cat = category_column.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"boxplot_{safe_num}_by_{safe_cat}.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return {"plot_path": out_path, "numeric_column": numeric_column, "category_column": category_column}


def _test_normality(csv_path: str, column: str) -> dict:
    df = _load(csv_path)
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric."}

    series = df[column].dropna()
    if len(series) < 3:
        return {"error": "Need at least 3 non-null values to test normality."}

    if len(series) < 5000:
        stat, p_value = scipy_stats.shapiro(series)
        test_used = "Shapiro-Wilk"
    else:
        stat, p_value = scipy_stats.normaltest(series)
        test_used = "D'Agostino-Pearson"

    is_normal = bool(p_value > 0.05)
    return {
        "column": column,
        "test": test_used,
        "statistic": round(float(stat), 6),
        "p_value": round(float(p_value), 6),
        "is_normal": is_normal,
        "interpretation": (
            f"{column} appears normally distributed (p={p_value:.4f} > 0.05)."
            if is_normal else
            f"{column} is likely NOT normally distributed (p={p_value:.4f} ≤ 0.05)."
        ),
    }


def _plot_pairplot(csv_path: str) -> dict:
    df = _load(csv_path)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) < 2:
        return {"error": "Need at least 2 numeric columns for a pairplot."}

    # Cap at 6 numeric columns to keep the plot readable
    numeric_cols = numeric_cols[:6]

    # Find first categorical column to use as hue
    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and df[col].dtype == object and df[col].nunique() <= 15
    ]
    hue_col = categorical_cols[0] if categorical_cols else None

    plot_df = df[numeric_cols + ([hue_col] if hue_col else [])]
    pair_grid = sns.pairplot(plot_df, hue=hue_col, diag_kind="kde",
                             plot_kws={"alpha": 0.6}, height=2.2)
    pair_grid.figure.suptitle("Pairwise Relationships", y=1.02)

    out_path = os.path.join(OUTPUT_DIR, "pairplot.png")
    pair_grid.figure.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close("all")

    return {"plot_path": out_path, "columns_plotted": numeric_cols, "hue": hue_col}


def _compute_data_quality_score(csv_path: str) -> dict:
    df = _load(csv_path)

    # Completeness: fraction of non-null values across entire dataframe
    total_cells = df.size
    missing_cells = int(df.isnull().sum().sum())
    completeness = round((1 - missing_cells / total_cells) * 100, 2) if total_cells else 100.0

    # Uniqueness: fraction of non-duplicate rows
    duplicate_count = int(df.duplicated().sum())
    uniqueness = round((1 - duplicate_count / len(df)) * 100, 2) if len(df) else 100.0

    # Consistency: penalise zero/near-zero variance columns and mixed-type columns
    issues = []
    consistency_penalties = 0

    for col in df.columns:
        # Near-zero variance numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 1:
                issues.append({"column": col, "issue": "constant column (zero variance)", "severity": "high"})
                consistency_penalties += 20
            elif df[col].std() < 1e-6:
                issues.append({"column": col, "issue": "near-zero variance", "severity": "medium"})
                consistency_penalties += 10

        # Mixed types in object columns (some values parseable as numbers, others not)
        if df[col].dtype == object:
            sample = df[col].dropna().head(100)
            numeric_like = sample.apply(lambda x: str(x).replace(".", "", 1).lstrip("-").isdigit())
            ratio = numeric_like.mean()
            if 0 < ratio < 0.8:
                issues.append({"column": col, "issue": f"mixed types ({ratio:.0%} numeric-like)", "severity": "high"})
                consistency_penalties += 15

    consistency_penalties = min(consistency_penalties, 100)
    consistency = round(max(0.0, 100.0 - consistency_penalties), 2)

    # Null-per-column issues
    null_pct = df.isnull().mean() * 100
    for col, pct in null_pct.items():
        if pct > 50:
            issues.append({"column": col, "issue": f"{pct:.1f}% missing values", "severity": "high"})
        elif pct > 10:
            issues.append({"column": col, "issue": f"{pct:.1f}% missing values", "severity": "medium"})

    if duplicate_count > 0:
        issues.append({
            "column": "—",
            "issue": f"{duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.1f}%)",
            "severity": "medium" if duplicate_count / len(df) < 0.1 else "high",
        })

    # Overall score: weighted average
    overall = round(completeness * 0.4 + uniqueness * 0.3 + consistency * 0.3, 1)

    return {
        "overall_score": overall,
        "breakdown": {
            "completeness": completeness,
            "uniqueness": uniqueness,
            "consistency": consistency,
        },
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_count,
        "issues": issues,
    }


def _detect_duplicates(csv_path: str) -> dict:
    df = _load(csv_path)
    mask = df.duplicated(keep=False)
    duplicate_count = int(df.duplicated().sum())
    pct = round(duplicate_count / len(df) * 100, 2) if len(df) else 0.0

    examples = []
    if duplicate_count:
        examples = df[mask].head(5).to_dict(orient="records")

    return {
        "duplicate_rows": duplicate_count,
        "duplicate_pct": pct,
        "total_rows": len(df),
        "examples": examples,
    }


def _plot_missing_heatmap(csv_path: str) -> dict:
    df = _load(csv_path)
    missing_counts = df.isnull().sum()

    if missing_counts.sum() == 0:
        return {"message": "No missing values found — heatmap not generated."}

    # Keep only columns that have at least one missing value
    cols_with_nulls = missing_counts[missing_counts > 0].index.tolist()
    plot_df = df[cols_with_nulls]

    # Cap rows for readability
    max_rows = 200
    if len(plot_df) > max_rows:
        plot_df = plot_df.iloc[:max_rows]
        truncated = True
    else:
        truncated = False

    fig, ax = plt.subplots(figsize=(max(6, len(cols_with_nulls) * 0.8), min(10, len(plot_df) * 0.06 + 2)))
    sns.heatmap(plot_df.isnull(), cbar=False, yticklabels=False,
                cmap=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title(f"Missing Value Map{' (first 200 rows)' if truncated else ''}\n(red = missing, green = present)")
    ax.set_xlabel("Columns")
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "missing_heatmap.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return {"plot_path": out_path, "columns_with_missing": cols_with_nulls, "truncated": truncated}


def _plot_qq(csv_path: str, column: str) -> dict:
    df = _load(csv_path)
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numeric."}

    series = df[column].dropna()
    if len(series) < 3:
        return {"error": "Need at least 3 non-null values for a Q-Q plot."}

    fig, ax = plt.subplots(figsize=(5, 5))
    (osm, osr), (slope, intercept, r) = scipy_stats.probplot(series, dist="norm")
    ax.plot(osm, osr, "o", alpha=0.5, markersize=4, color="steelblue", label="Data")
    ax.plot(osm, slope * osm + intercept, "r-", linewidth=1.5, label="Normal fit")
    ax.set_title(f"Q-Q Plot: {column}\n(R²={r**2:.3f})")
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.legend(fontsize=8)
    fig.tight_layout()

    safe_col = column.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(OUTPUT_DIR, f"qq_{safe_col}.png")
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    return {"plot_path": out_path, "column": column, "r_squared": round(r ** 2, 4)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
