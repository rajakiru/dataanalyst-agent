"""
MCP Server: Data Analysis Tools
Exposes 15 tools via stdio transport for the collection agent to call.
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

# Image processing
try:
    from image_processor import ImageProcessor, process_images_to_csv
    IMAGE_SUPPORT_AVAILABLE = True
except ImportError:
    IMAGE_SUPPORT_AVAILABLE = False
    print("Warning: image_processor not available. Image processing tool will be disabled.", file=sys.stderr)

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
                "(numeric, categorical, datetime, or id). "
                "USE CASE: Always run first to understand data structure and inform downstream analysis choices. "
                "ISSUES DETECTED: Missing values, incorrect type conversions, ID columns, mixed-type columns."
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
                "quartiles, skewness, and kurtosis. "
                "USE CASE: Identify data distribution shape, central tendency, and spread. Detect skewed distributions. "
                "ISSUES DETECTED: High skewness (>|1|) indicates outliers or non-normal distributions; high kurtosis indicates heavy tails. "
                "SOLUTIONS: Use Box-Cox transformation for skewed data, or stratify analysis by skewness."
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
                "Returns the matrix as a nested dict and lists the top 5 strongest correlations. "
                "USE CASE: Identify redundant features, multicollinearity risks, and potential predictive relationships. "
                "ISSUES DETECTED: Very high correlations (>0.9) suggest redundancy; multicollinearity risks. "
                "SOLUTIONS: Consider feature engineering, dropping redundant columns, or applying dimensionality reduction (PCA)."
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
                "Returns the count of outliers, their indices, and the IQR bounds used. "
                "USE CASE: Identify potential data entry errors, measurement noise, or rare valid events. "
                "ISSUES DETECTED: >5% outliers suggest possible data quality issues or skewed distributions. "
                "SOLUTIONS: Investigate root cause, remove if errors, use Winsorization if valid, or apply robust statistics."
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
                "USE CASE: Visually assess normality, modality, skewness, and outlier presence. "
                "ISSUES DETECTED: Multimodal distributions, long tails, gaps, bimodality. "
                "SOLUTIONS: For categorical: investigate value imbalance; for numeric: consider transformations or stratified analysis."
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
                "USE CASE: Identify multicollinearity, redundant features, and feature interaction patterns at a glance. "
                "ISSUES DETECTED: Strong clusters of correlated features, near-zero correlations. "
                "SOLUTIONS: Use feature selection, PCA, or domain knowledge to prune redundant features."
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
                "USE CASE: Visualize bivariate relationships, linearity, and outlier influence. "
                "Run only for column pairs with |correlation| > 0.5 to avoid noise. "
                "ISSUES DETECTED: Weak regression fit (scattered points), outliers, non-linear relationships. "
                "SOLUTIONS: Use polynomial features, non-linear regression, or robust regression if outliers present."
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
                "USE CASE: Identify group-wise outliers, imbalanced group distributions, and homogeneity violations. "
                "ISSUES DETECTED: Outliers within groups, vastly different variances across groups, group imbalance. "
                "SOLUTIONS: Investigate why groups differ; consider stratified analysis or weighted models; address group imbalance via resampling."
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
                "5000 rows, otherwise D'Agostino-Pearson. Returns the test statistic, p-value. "
                "USE CASE: Validate assumptions for parametric tests (t-tests, ANOVA). Inform transformation strategy. "
                "ISSUES DETECTED: p < 0.05 indicates non-normal distribution. "
                "SOLUTIONS: Apply log, Box-Cox, or reciprocal transformations; use non-parametric tests; stratify by skewness."
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
                "USE CASE: Holistic view of all bivariate relationships, identify clustering or separability. "
                "ISSUES DETECTED: Non-linear relationships, weak separation between groups, outliers in multiple dimensions."
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
                "Compute an overall data quality score (0-100) and breakdown: "
                "completeness (non-null %), uniqueness (non-duplicate %), consistency (type uniformity). "
                "USE CASE: Assess dataset fitness for downstream analysis. Prioritize data cleaning efforts. "
                "ISSUES DETECTED: Low completeness (missing data), duplicates, type inconsistencies, zero-variance columns. "
                "SOLUTIONS: See recommend_solutions tool for per-issue remediation strategies."
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
                "Detect duplicate rows in the dataset. Returns the count, percentage, and examples. "
                "USE CASE: Identify data integrity issues, prevent data leakage into train/test splits. "
                "ISSUES DETECTED: >0.5% duplicates suggest possible data entry or integration errors. "
                "SOLUTIONS: Remove exact duplicates; investigate if partial duplicates represent real phenomena or errors."
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
                "USE CASE: Visualize missing data patterns (MCAR, MAR, MNAR), identify columns to drop or impute. "
                "ISSUES DETECTED: Systematic missing patterns, high-missingness columns. "
                "SOLUTIONS: Use MCAR tests for imputation strategy; consider mean/KNN/forward-fill for MAR; drop MNAR columns if >50% missing."
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
                "normality. "
                "USE CASE: Visual complement to statistical normality tests (test_normality). "
                "ISSUES DETECTED: Deviations from diagonal line indicate non-normality. Curved tails = skewness, bent = kurtosis. "
                "SOLUTIONS: Apply appropriate transformations (log for right-skew, Box-Cox for general skew)."
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
        Tool(
            name="recommend_solutions",
            description=(
                "Generate actionable remediation recommendations for data quality issues, distributional anomalies, and analysis challenges. "
                "Takes a list of detected issues (from data quality agent, normality tests, anomaly detection, etc.) "
                "and returns concrete, prioritized solutions with implementation guidance. "
                "USE CASE: Bridge the gap between problem detection and solution implementation. "
                "Ensures every detected issue receives a tailored fix recommendation."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_path": {"type": "string", "description": "Absolute path to the CSV file."},
                    "issues": {
                        "type": "array",
                        "description": (
                            "List of detected issues to generate solutions for. Each issue should be a dict with: "
                            "column (str), issue_type (str from: 'missing_values', 'duplicates', 'outliers', 'non_normal', "
                            "'type_inconsistency', 'high_cardinality', 'zero_variance', 'multicollinearity', 'imbalanced_groups'), "
                            "severity (str: 'low', 'medium', 'high'), details (any additional context)."
                        ),
                        "items": {"type": "object"},
                    },
                },
                "required": ["csv_path", "issues"],
            },
        ),
        Tool(
            name="process_images_to_csv",
            description=(
                "Process a directory of images into a CSV feature table for analysis. "
                "Extracts features from each image (using pre-trained CNN or fallback color histograms) and metadata (EXIF, file stats). "
                "Returns a CSV file with image features and metadata, ready for downstream tabular analysis. "
                "USE CASE: Convert image datasets (Iris flowers, product photos, etc.) into tabular features for numerical analysis. "
                "ARCHITECTURE: Parallel per-image error handling - failures in individual images don't block the entire dataset. "
                "COVERAGE: Reports % of images successfully processed. Error budget: 5-20% failures is acceptable. "
                "OUTPUT: Returns path to generated CSV + metadata dict with coverage metrics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_dir": {"type": "string", "description": "Absolute path to directory containing images (JPEG, PNG, BMP, etc.)."},
                    "output_csv": {"type": "string", "description": "Optional: output CSV path. Auto-generated if not provided. Default: outputs/images_features_<timestamp>.csv"},
                },
                "required": ["image_dir"],
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
    elif name == "recommend_solutions":
        result = _recommend_solutions(arguments["csv_path"], arguments["issues"])
    elif name == "process_images_to_csv":
        output_csv = arguments.get("output_csv", None)
        result = _process_images_to_csv(arguments["image_dir"], output_csv)
    else:
        result = {"error": f"Unknown tool: {name}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------

def _load(csv_path: str) -> pd.DataFrame:
    """Load CSV with basic validation."""
    df = pd.read_csv(csv_path)
    
    # Guard: empty dataframe
    if df.empty:
        raise ValueError("CSV file is empty (0 rows or 0 columns)")
    
    # Guard: too many columns (avoid memory explosion in visualizations)
    if df.shape[1] > 200:
        raise ValueError(f"CSV has {df.shape[1]} columns. Max supported: 200.")
    
    # Guard: too many rows (avoid timeout/memory)
    if df.shape[0] > 1_000_000:
        raise ValueError(f"CSV has {df.shape[0]} rows. Max supported: 1,000,000.")
    
    return df


def _infer_schema(csv_path: str) -> dict:
    df = _load(csv_path)
    
    # Guard: single column dataset
    if df.shape[1] == 1:
        col = df.columns[0]
        roles = {col: "unknown"}
        return {
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": {col: int(n) for col, n in df.isnull().sum().items()},
            "null_pct": {col: round(float(n) / len(df) * 100, 2) for col, n in df.isnull().sum().items()},
            "roles": roles,
            "sample_values": {col: df[col].dropna().head(3).tolist() for col in df.columns},
            "note": "Dataset has only 1 column. Correlation and multivariate analysis not applicable.",
        }
    
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
        "null_pct": {col: round(float(n) / len(df) * 100, 2) for col, n in df.isnull().sum().items()},
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
    if len(series) < 3:
        return {"error": "Need at least 3 non-null values for anomaly detection."}
    
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    
    # Guard: if IQR is zero (low-variance or constant column), skip outlier detection
    if iqr < 1e-10:
        return {
            "column": column,
            "iqr_bounds": {"lower": round(float(q1), 4), "upper": round(float(q3), 4)},
            "q1": round(float(q1), 4),
            "q3": round(float(q3), 4),
            "iqr": 0.0,
            "outlier_count": 0,
            "outlier_pct": 0.0,
            "outlier_indices": [],
            "note": "IQR is zero (constant or near-constant column); outlier detection skipped.",
        }

    # Adapt multiplier to reduce anchoring bias on the conventional 1.5x.
    # Highly skewed columns produce many false positives at 1.5x; widen to 3x.
    # Near-symmetric columns are better served by the tighter 1.5x threshold.
    skewness = float(series.skew())
    iqr_multiplier = 3.0 if abs(skewness) > 2 else 2.0 if abs(skewness) > 1 else 1.5

    lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr

    outlier_mask = (df[column] < lower) | (df[column] > upper)
    outlier_indices = df.index[outlier_mask].tolist()

    return {
        "column": column,
        "iqr_bounds": {"lower": round(float(lower), 4), "upper": round(float(upper), 4)},
        "q1": round(float(q1), 4),
        "q3": round(float(q3), 4),
        "iqr": round(float(iqr), 4),
        "iqr_multiplier": iqr_multiplier,
        "skewness": round(skewness, 4),
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
    
    # Guard: handle empty dataframe
    if df.empty:
        return {
            "overall_score": 0.0,
            "breakdown": {"completeness": 0.0, "uniqueness": 0.0, "consistency": 0.0},
            "missing_cells": 0,
            "duplicate_rows": 0,
            "issues": [{"column": "—", "issue": "Dataset is empty", "severity": "high"}],
        }

    # Completeness: fraction of non-null values across entire dataframe
    total_cells = df.size
    missing_cells = int(df.isnull().sum().sum())
    completeness = round((1 - missing_cells / total_cells) * 100, 2) if total_cells > 0 else 100.0

    # Uniqueness: fraction of non-duplicate rows
    duplicate_count = int(df.duplicated().sum())
    uniqueness = round((1 - duplicate_count / len(df)) * 100, 2) if len(df) > 0 else 100.0

    # Consistency: penalise zero/near-zero variance columns and mixed-type columns
    issues = []
    consistency_penalties = 0

    for col in df.columns:
        # Near-zero variance numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            std_val = df[col].std()
            if df[col].nunique() <= 1:
                issues.append({"column": col, "issue": "constant column (zero variance)", "severity": "high"})
                consistency_penalties += 20
            elif pd.isna(std_val) or std_val < 1e-6:
                issues.append({"column": col, "issue": "near-zero variance", "severity": "medium"})
                consistency_penalties += 10

        # Mixed types in object columns (some values parseable as numbers, others not)
        if df[col].dtype == object:
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
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
        dup_pct = duplicate_count / len(df) * 100 if len(df) > 0 else 0
        issues.append({
            "column": "—",
            "issue": f"{duplicate_count} duplicate rows ({dup_pct:.1f}%)",
            "severity": "medium" if dup_pct < 10 else "high",
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
    pct = round(duplicate_count / len(df) * 100, 2) if len(df) > 0 else 0.0

    examples = []
    if duplicate_count:
        # Cap at 10 examples to avoid overwhelming UI/reports
        examples = df[mask].head(10).to_dict(orient="records")

    return {
        "duplicate_rows": duplicate_count,
        "duplicate_pct": pct,
        "total_rows": len(df),
        "examples": examples,
        "truncated": duplicate_count > 10,
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


def _recommend_solutions(csv_path: str, issues: list) -> dict:
    """
    Generate actionable solutions for each detected issue.
    Returns a prioritized list of recommendations with implementation guidance.
    """
    if not isinstance(issues, list) or len(issues) == 0:
        return {
            "total_issues": 0,
            "solutions_provided": 0,
            "recommendations": [],
            "summary": "No issues detected — data quality is excellent!"
        }

    df = _load(csv_path)
    solutions = []

    for issue in issues:
        if not isinstance(issue, dict):
            continue

        column = issue.get("column", "—")
        issue_type = issue.get("issue_type", "unknown")
        severity = issue.get("severity", "medium")
        details = issue.get("details") or {}  # guard: details key may exist with None value

        recommendation = {
            "column": column,
            "issue_type": issue_type,
            "severity": severity,
            "detected_value": details.get("value"),
            "actions": [],
        }

        # ====== MISSING VALUES ======
        if issue_type == "missing_values":
            missing_pct = details.get("missing_pct", 0)

            if missing_pct > 50:
                recommendation["actions"] = [
                    {
                        "priority": "high",
                        "action": "Consider dropping the column",
                        "rationale": f"Column has {missing_pct}% missing data, making it unreliable for analysis.",
                        "implementation": f"df = df.drop('{column}', axis=1)"
                    }
                ]
            else:
                col_type = "numeric" if column in df.columns and pd.api.types.is_numeric_dtype(df[column]) else "categorical"

                if col_type == "numeric":
                    recommendation["actions"] = [
                        {
                            "priority": "high",
                            "action": "Impute with mean/median (parametric approach)",
                            "rationale": "Preserves distributional properties for numeric data.",
                            "implementation": f"df['{column}'] = df['{column}'].fillna(df['{column}'].median())"
                        },
                        {
                            "priority": "medium",
                            "action": "Impute with KNN (for correlated features)",
                            "rationale": "Better for complex relationships between features.",
                            "implementation": "from sklearn.impute import KNNImputer; imputer = KNNImputer(); df = imputer.fit_transform(df)"
                        },
                        {
                            "priority": "low",
                            "action": "Create a 'missing' indicator feature",
                            "rationale": "If missingness is informative for downstream models.",
                            "implementation": f"df['{column}_is_missing'] = df['{column}'].isnull().astype(int)"
                        }
                    ]
                else:
                    recommendation["actions"] = [
                        {
                            "priority": "high",
                            "action": "Impute with mode (most frequent value)",
                            "rationale": "Categorical data: preserve class distribution.",
                            "implementation": f"df['{column}'] = df['{column}'].fillna(df['{column}'].mode()[0])"
                        },
                        {
                            "priority": "medium",
                            "action": "Create 'Unknown' category",
                            "rationale": "If missingness is informative.",
                            "implementation": f"df['{column}'] = df['{column}'].fillna('Unknown')"
                        }
                    ]

        # ====== DUPLICATES ======
        elif issue_type == "duplicates":
            duplicate_pct = details.get("duplicate_pct", 0)
            recommendation["actions"] = [
                {
                    "priority": "high",
                    "action": "Remove exact duplicates",
                    "rationale": f"Duplicates ({duplicate_pct}%) inflate sample size and introduce bias.",
                    "implementation": "df = df.drop_duplicates()"
                },
                {
                    "priority": "medium",
                    "action": "Investigate root cause",
                    "rationale": "Understand if duplicates are data entry errors or legitimate repeated observations.",
                    "implementation": "df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head(10)"
                }
            ]

        # ====== OUTLIERS ======
        elif issue_type == "outliers":
            outlier_pct = details.get("outlier_pct", 0)
            skewness = details.get("skewness", 0)
            iqr_multiplier = details.get("iqr_multiplier", 1.5)

            # Action bias guard: if the outlier detection already used a widened multiplier
            # due to high skewness, the "outliers" are likely the legitimate tail of the
            # distribution — flag this rather than recommending removal.
            if abs(skewness) > 2 and iqr_multiplier >= 3.0:
                recommendation["actions"] = [
                    {
                        "priority": "low",
                        "action": "No remediation recommended",
                        "rationale": (
                            f"Column '{column}' is highly skewed (skewness={skewness:.2f}). "
                            "Outliers detected at the widened 3×IQR threshold are likely the "
                            "natural tail of a skewed distribution, not data errors. "
                            "Consider a log or Box-Cox transform before re-evaluating."
                        ),
                        "implementation": f"df['{column}_log'] = np.log1p(df['{column}'])  # then re-run anomaly detection"
                    }
                ]
            else:
                recommendation["actions"] = [
                    {
                        "priority": "high" if outlier_pct > 5 else "medium",
                        "action": "Investigate outlier sources",
                        "rationale": "Determine if outliers are measurement errors, legitimate extremes, or separate phenomena.",
                        "implementation": "df[df['%s'].isin([...])]  # inspect specific outlier rows" % column
                    },
                    {
                        "priority": "medium",
                        "action": "Winsorize outliers (capping strategy)",
                        "rationale": "Reduces outlier influence while retaining data.",
                        "implementation": "from scipy.stats.mstats import winsorize; df['%s'] = winsorize(df['%s'], limits=[0.05, 0.05])" % (column, column)
                    },
                    {
                        "priority": "medium" if outlier_pct < 5 else "low",
                        "action": "Remove outliers (deletion strategy)",
                        "rationale": "Only if confident outliers are errors.",
                        "implementation": "df = df[(df['%s'] >= lower_bound) & (df['%s'] <= upper_bound)]" % (column, column)
                    }
                ]

                if abs(skewness) > 1:
                    recommendation["actions"].append({
                        "priority": "medium",
                        "action": "Apply transformation (Box-Cox or log)",
                        "rationale": f"High skewness ({skewness:.2f}) indicates outliers may be systematic; transformation can normalize.",
                        "implementation": "from scipy.stats import boxcox; df['%s_transformed'] = boxcox(df['%s'] + 1)[0]" % (column, column)
                    })

        # ====== NON-NORMAL DISTRIBUTION ======
        elif issue_type == "non_normal":
            skewness = details.get("skewness", 0)

            recommendation["actions"] = [
                {
                    "priority": "high",
                    "action": "Apply log transformation",
                    "rationale": "Effective for right-skewed data (common in real-world metrics).",
                    "implementation": "df['%s_log'] = np.log1p(df['%s'])" % (column, column)
                },
                {
                    "priority": "high",
                    "action": "Apply Box-Cox transformation",
                    "rationale": "Optimal power transformation for normality; works for non-negative data.",
                    "implementation": "from scipy.stats import boxcox; df['%s_transformed'], lambda_param = boxcox(df['%s'] + 1)" % (column, column)
                },
                {
                    "priority": "medium",
                    "action": "Use non-parametric tests",
                    "rationale": "If parametric assumptions are critical, switch to rank-based tests (Mann-Whitney, Kruskal-Wallis).",
                    "implementation": "from scipy.stats import mannwhitneyu; stat, p = mannwhitneyu(group1, group2)"
                },
                {
                    "priority": "low",
                    "action": "Stratify analysis by subgroups",
                    "rationale": "Distribution may be normal within subgroups.",
                    "implementation": "df.groupby('category').apply(lambda x: x['%s'].describe())" % column
                }
            ]

        # ====== TYPE INCONSISTENCY ======
        elif issue_type == "type_inconsistency":
            mixed_ratio = details.get("mixed_ratio", 0)
            recommendation["actions"] = [
                {
                    "priority": "high",
                    "action": "Standardize data types",
                    "rationale": f"Mixed types ({mixed_ratio}% numeric-like) prevent proper analysis.",
                    "implementation": "df['%s'] = pd.to_numeric(df['%s'], errors='coerce')" % (column, column)
                },
                {
                    "priority": "high",
                    "action": "Investigate and handle exceptions",
                    "rationale": "Understand why non-numeric values exist; decide if they should be dropped or recoded.",
                    "implementation": "df[~df['%s'].apply(lambda x: str(x).isnumeric())]" % column
                }
            ]

        # ====== HIGH CARDINALITY ======
        elif issue_type == "high_cardinality":
            unique_count = details.get("unique_count", 0)
            recommendation["actions"] = [
                {
                    "priority": "high",
                    "action": "Group rare categories",
                    "rationale": f"Too many categories ({unique_count}) dilutes signal; group infrequent ones.",
                    "implementation": "rare_threshold = 0.02; rare = df['%s'].value_counts() < len(df) * rare_threshold; df.loc[df['%s'].isin(rare[rare].index), '%s'] = 'Other'" % (column, column, column)
                },
                {
                    "priority": "medium",
                    "action": "Apply one-hot encoding selectively",
                    "rationale": "Encode top-N categories only; recode rest as 'Other'.",
                    "implementation": "top_n = 10; top_cats = df['%s'].value_counts().head(top_n).index; df['%s'] = df['%s'].apply(lambda x: x if x in top_cats else 'Other')" % (column, column, column)
                },
                {
                    "priority": "low",
                    "action": "Use target encoding",
                    "rationale": "For supervised learning; encode categories by target mean.",
                    "implementation": "target_means = df.groupby('%s')['target'].mean(); df['%s_encoded'] = df['%s'].map(target_means)" % (column, column, column)
                }
            ]

        # ====== ZERO VARIANCE ======
        elif issue_type == "zero_variance":
            recommendation["actions"] = [
                {
                    "priority": "high",
                    "action": "Drop the column",
                    "rationale": "Constant value provides no signal for analysis or modeling.",
                    "implementation": f"df = df.drop('{column}', axis=1)"
                }
            ]

        # ====== MULTICOLLINEARITY ======
        elif issue_type == "multicollinearity":
            corr_value = details.get("correlation", 0.9)
            corr_partner = details.get("correlated_with", "unknown")
            recommendation["actions"] = [
                {
                    "priority": "high",
                    "action": "Drop redundant column",
                    "rationale": f"Highly correlated with {corr_partner} ({corr_value:.3f}); dropping reduces noise and improves interpretability.",
                    "implementation": f"df = df.drop('{column}', axis=1)"
                },
                {
                    "priority": "medium",
                    "action": "Apply PCA (Principal Component Analysis)",
                    "rationale": "Combine correlated features into uncorrelated components.",
                    "implementation": "from sklearn.decomposition import PCA; pca = PCA(); df_pca = pca.fit_transform(df[numeric_cols])"
                },
                {
                    "priority": "low",
                    "action": "Use regularized regression (L1/L2)",
                    "rationale": "Automatic feature selection handles multicollinearity.",
                    "implementation": "from sklearn.linear_model import Ridge; model = Ridge(alpha=1.0).fit(X, y)"
                }
            ]

        # ====== IMBALANCED GROUPS ======
        elif issue_type == "imbalanced_groups":
            recommendation["actions"] = [
                {
                    "priority": "high",
                    "action": "Stratified sampling in train/test splits",
                    "rationale": "Ensures train/test maintain class distribution.",
                    "implementation": "from sklearn.model_selection import train_test_split; train_test_split(..., stratify=y)"
                },
                {
                    "priority": "high",
                    "action": "Use class weights in models",
                    "rationale": "Give more weight to minority class during training.",
                    "implementation": "from sklearn.utils.class_weight import compute_class_weight; weights = compute_class_weight('balanced', np.unique(y), y)"
                },
                {
                    "priority": "medium",
                    "action": "Oversample minority / Undersample majority",
                    "rationale": "Rebalance class distribution if acceptable.",
                    "implementation": "from imblearn.over_sampling import SMOTE; smote = SMOTE(); X_resampled, y_resampled = smote.fit_resample(X, y)"
                },
                {
                    "priority": "low",
                    "action": "Use appropriate metrics (precision, recall, F1)",
                    "rationale": "Accuracy is misleading for imbalanced data; prefer ROC-AUC or F1-score.",
                    "implementation": "from sklearn.metrics import classification_report, roc_auc_score"
                }
            ]

        if recommendation["actions"]:
            solutions.append(recommendation)

    return {
        "total_issues": len(issues),
        "solutions_provided": len(solutions),
        "recommendations": solutions,
        "summary": (
            f"Generated {len(solutions)} actionable recommendations for {len(issues)} detected issues. "
            f"Prioritize 'high' actions first; 'medium' and 'low' are alternatives or optimizations."
        )
    }


def _process_images_to_csv(image_dir: str, output_csv: str = None) -> dict:
    """
    Process images directory to CSV feature table.
    
    Extracts features from all images using pre-trained CNN or fallback color histograms.
    Returns path to generated CSV + metadata with coverage metrics.
    """
    if not IMAGE_SUPPORT_AVAILABLE:
        return {
            "error": "Image processing not available. Install torchvision and Pillow.",
            "available": False
        }
    
    try:
        # Validate image directory
        image_dir_path = os.path.abspath(image_dir)
        if not os.path.isdir(image_dir_path):
            return {
                "error": f"Image directory not found: {image_dir_path}",
                "available": True
            }
        
        # Set output path
        if output_csv is None:
            from datetime import datetime
            output_csv = os.path.join(OUTPUT_DIR, f"images_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        else:
            output_csv = os.path.abspath(output_csv)
        
        # Process images
        df_features, metadata = process_images_to_csv(image_dir_path, output_csv)
        
        return {
            "success": True,
            "output_csv": output_csv,
            "processed_count": metadata["processed_count"],
            "failed_count": metadata["failed_count"],
            "total_count": metadata["total_count"],
            "coverage_percent": round(metadata["coverage_percent"], 1),
            "feature_dimension": metadata["feature_dimension"],
            "feature_count": len(df_features),
            "columns_generated": list(df_features.columns),
            "errors": metadata.get("errors", [])[:5],  # First 5 errors
            "note": (
                f"Successfully processed {metadata['processed_count']}/{metadata['total_count']} images "
                f"({metadata['coverage_percent']:.1f}% coverage). "
                f"Features saved to {output_csv}. "
                "Ready for downstream tabular analysis (schema, statistics, anomalies)."
            )
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "available": True,
            "error_type": type(e).__name__
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
