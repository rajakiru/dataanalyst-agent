"""
Agent orchestration: collection agent + reporting agent.
Uses Dedalus (OpenAI-compatible) for LLM calls and MCP stdio for tool execution.
"""

import asyncio
import glob
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

MCP_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server.py")

# Tool names grouped by UI category so the pipeline can be filtered per user selection.
_ALWAYS_TOOLS = frozenset({"infer_schema", "summarize_statistics", "compute_correlations"})
_VIZ_TOOLS = frozenset({
    "plot_pairplot", "plot_correlation_heatmap", "plot_scatter",
    "plot_boxplot", "plot_distribution",
})
_ANOMALY_TOOLS = frozenset({"detect_anomalies"})
_QUALITY_TOOLS = frozenset({
    "compute_data_quality_score", "detect_duplicates",
    "test_normality", "plot_missing_heatmap", "plot_qq",
})
_SOLUTIONS_TOOLS = frozenset({"recommend_solutions"})
_ALL_CATEGORIES = frozenset({"visualizations", "anomaly_detection", "data_quality", "solutions"})


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------

def _mcp_tool_to_openai(tool) -> dict:
    """Convert an MCP Tool object to an OpenAI function-calling spec."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        },
    }


def _map_issue_string_to_type(issue_str: str) -> str:
    """Map issue description strings to standardized issue types for solutions agent."""
    issue_lower = issue_str.lower()
    
    if "missing" in issue_lower or "null" in issue_lower:
        return "missing_values"
    elif "duplicate" in issue_lower:
        return "duplicates"
    elif "outlier" in issue_lower or "anomal" in issue_lower:
        return "outliers"
    elif "normal" in issue_lower or "distribution" in issue_lower:
        return "non_normal"
    elif "type" in issue_lower or "mixed" in issue_lower:
        return "type_inconsistency"
    elif "constant" in issue_lower or "variance" in issue_lower or "zero" in issue_lower:
        return "zero_variance"
    elif "cardinality" in issue_lower or "unique" in issue_lower:
        return "high_cardinality"
    elif "correlation" in issue_lower or "collinearity" in issue_lower:
        return "multicollinearity"
    elif "imbalance" in issue_lower or "group" in issue_lower:
        return "imbalanced_groups"
    else:
        return "unknown"


def _detect_and_process_images(input_path: str, model: str, api_key: str = "") -> tuple[str, dict]:
    """
    Detect if input is an image directory and process it to CSV.
    
    Returns: (csv_path, process_metadata)
    If input is CSV, returns (csv_path, {})
    If input is image dir, processes images and returns (generated_csv_path, metadata)
    """
    input_path = os.path.abspath(input_path)
    
    # Check if it's a CSV file
    if os.path.isfile(input_path) and input_path.lower().endswith('.csv'):
        # Peek at columns — if it has feature_* columns it's an image features CSV
        # and should be treated as image_dataset so the right agent prompts are used.
        try:
            import pandas as _pd
            _cols = _pd.read_csv(input_path, nrows=0).columns.tolist()
            _feat_count = sum(1 for c in _cols if c.startswith("feature_"))
            if _feat_count >= 16:
                _n_rows = sum(1 for _ in open(input_path)) - 1
                _nan_rows = int(_pd.read_csv(input_path)[
                    [c for c in _cols if c.startswith("feature_")]
                ].isnull().all(axis=1).sum())
                _processed = _n_rows - _nan_rows
                return input_path, {
                    "processed_count":  _processed,
                    "failed_count":     _nan_rows,
                    "total_count":      _n_rows,
                    "coverage_percent": round(_processed / _n_rows * 100, 1) if _n_rows else 100,
                    "feature_dimension": _feat_count,
                    "errors": [],
                    "source": "precomputed_csv",
                }
        except Exception:
            pass
        return input_path, {}
    
    # Check if it's a directory with images
    if os.path.isdir(input_path):
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in os.listdir(input_path) 
                      if os.path.splitext(f)[1].lower() in image_extensions]
        
        if image_files:
            print(f"\n🖼️  Image dataset detected: {len(image_files)} images in {input_path}")
            print(f"    Processing images → features → CSV...")
            
            # Run image processing synchronously
            import sys
            sys.path.insert(0, os.path.dirname(__file__))
            try:
                from image_processor import process_images_to_csv as img_to_csv
                
                outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
                os.makedirs(outputs_dir, exist_ok=True)
                from datetime import datetime as _dt
                out_csv = os.path.join(outputs_dir, f"images_features_{_dt.now().strftime('%Y%m%d_%H%M%S')}.csv")
                csv_path, metadata = img_to_csv(input_path, out_csv)
                print(f"    ✓ Processed {metadata['processed_count']}/{metadata['total_count']} images ({metadata['coverage_percent']:.1f}% coverage)")
                print(f"    ✓ Generated CSV: {csv_path}")
                print(f"    Running tabular analysis pipeline on {os.path.getsize(csv_path)} bytes...\n")
                
                return csv_path, metadata
            except Exception as e:
                print(f"    ✗ Image processing failed: {e}")
                raise
    
    # Not a CSV, not an image directory
    raise ValueError(f"Input must be a CSV file or directory of images: {input_path}")


async def _run_pipeline_async(csv_path: str, model: str, api_key: str = "", source_type: str = "csv_dataset", enabled_categories: frozenset | None = None) -> dict:
    """
    Full async pipeline:
    1. Spawn MCP server subprocess (stdio).
    2. Collection agent: LLM decides which tools to call, executes them.
    3. Reporting agent: LLM summarises all tool outputs into a narrative.
    Returns a dict with all results.
    """
    # Clear stale plots from previous runs before starting
    outputs_dir = os.path.join(os.path.dirname(MCP_SERVER_PATH), "outputs")
    for old_file in glob.glob(os.path.join(outputs_dir, "*.png")):
        try:
            os.remove(old_file)
        except OSError:
            pass

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_PATH],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools and convert to OpenAI format
            mcp_tools_response = await session.list_tools()
            mcp_tools = mcp_tools_response.tools
            openai_tools = [_mcp_tool_to_openai(t) for t in mcp_tools]

            # Resolve enabled categories (None → all enabled)
            _enabled = enabled_categories if enabled_categories is not None else _ALL_CATEGORIES

            # Tools available to the collection agent (schema/stats always; others gated)
            _collection_allowed = set(_ALWAYS_TOOLS)
            if "visualizations" in _enabled:
                _collection_allowed |= _VIZ_TOOLS
            if "anomaly_detection" in _enabled:
                _collection_allowed |= _ANOMALY_TOOLS
            collection_tools = [t for t in openai_tools if t["function"]["name"] in _collection_allowed]

            # ----------------------------------------------------------------
            # Collection Agent
            # ----------------------------------------------------------------
            effective_key = api_key or OPENAI_API_KEY
            client = OpenAI(api_key=effective_key, base_url=OPENAI_BASE_URL)

            _image_rules = """
IMAGE DATASET RULES (this CSV was derived from image feature extraction):
- Columns named feature_0…feature_N are abstract CNN/histogram embeddings — NOT interpretable individually.
- Do NOT run `plot_distribution` or `detect_anomalies` or `test_normality` on feature_N columns — they produce noise, not insight.
- DO run `summarize_statistics`, `compute_correlations`, `plot_correlation_heatmap`, and `plot_pairplot` on ALL columns — these reveal embedding structure.
- DO run `plot_distribution`, `detect_anomalies`, and `test_normality` on metadata columns only (filename, width_px, height_px, file_size_kb, aspect_ratio).
- For `plot_boxplot`: use a metadata numeric column (e.g., file_size_kb) grouped by any categorical column if present.
- Interpret findings in terms of image diversity, coverage gaps, and metadata anomalies — not raw numeric values.""" if source_type == "image_dataset" else ""

            system_prompt = f"""You are an expert data analyst agent. You have access to tools that
operate on CSV files. Your goal is to produce the most insightful analysis possible for the
specific dataset you are given — not to blindly run every tool.

The CSV file path for ALL tool calls is: {csv_path}

CRITICAL EDGE CASE HANDLING:
- If the schema inference returns an error (empty dataset, >200 columns, >1M rows), STOP and report the error clearly.
- If there are 0 numeric columns, skip correlation, normality, and anomaly detection.
- If there is only 1 column, skip multi-column analyses (correlation, boxplot, pairplot).
- If any tool call returns an error, log it and continue with remaining tools.
- Be defensive: assume tools may fail and plan for graceful fallbacks.

DECISION RULES — apply these based on what you observe in the data:
- Run `summarize_statistics` and `test_normality` for every numeric column found.
- Run `compute_correlations` if there are 2+ numeric columns.
- Run `detect_anomalies` for every numeric column.
- If a column has high skewness (>1 or <-1), note this when interpreting its anomaly results.
- Do NOT call a tool that is irrelevant to this dataset's structure.

VISUALIZATION BUDGET — generate at most 6 plots total. Quality over quantity:
- ALWAYS generate `plot_pairplot` if there are 2+ numeric columns (it covers individual distributions on its diagonal — do NOT also call `plot_distribution` for those columns).
- ALWAYS generate `plot_correlation_heatmap` if there are 2+ numeric columns.
- Generate `plot_scatter` for the top 2 most strongly correlated pairs only (|corr| > 0.5). Do not scatter-plot every correlated pair.
- Generate `plot_boxplot` for the single most analytically interesting numeric+categorical pairing only.
- Generate `plot_distribution` ONLY for columns the pairplot does NOT cover (e.g., a lone categorical column) OR for numeric columns with extreme skewness (>2) that warrant a standalone close-up.
- If the budget is reached, skip additional plots rather than exceeding 6.
{_image_rules}
You will first be asked to reason and plan. Then execute your plan."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please analyse this CSV file: {csv_path}"},
            ]

            tool_results: list[dict] = []
            plot_paths: list[str] = []
            analysis_plan: str = ""
            max_iterations = 40  # safety cap

            # ----------------------------------------------------------------
            # Phase 1: Force schema discovery first
            # ----------------------------------------------------------------
            schema_response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=collection_tools,
                tool_choice={"type": "function", "function": {"name": "infer_schema"}},
            )
            schema_msg = schema_response.choices[0].message
            messages.append(schema_msg.model_dump(exclude_unset=True))

            # Execute the forced infer_schema call
            if schema_msg.tool_calls:
                tc = schema_msg.tool_calls[0]
                fn_args = json.loads(tc.function.arguments)
                
                try:
                    mcp_result = await session.call_tool("infer_schema", fn_args)

                    result_text = "{}"
                    if mcp_result.content:
                        first = mcp_result.content[0]
                        if hasattr(first, "text") and first.text:
                            result_text = first.text

                    try:
                        result_dict = json.loads(result_text) if result_text.strip() else {}
                    except json.JSONDecodeError:
                        result_dict = {"error": result_text}
                    
                    # Guard: check for schema errors that indicate problematic datasets
                    if "error" in result_dict:
                        analysis_plan = f"Schema inference failed: {result_dict.get('error', 'Unknown error')}. Analysis cannot proceed."
                        tool_results.append({"tool": "infer_schema", "args": fn_args, "result": result_dict})
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                        messages.append({"role": "user", "content": "Schema inference returned an error. Please report this and stop analysis."})
                        
                        # Short-circuit: return early with error state
                        return {
                            "tool_results": tool_results,
                            "plot_paths": plot_paths,
                            "narrative": f"Pipeline Error: {result_dict.get('error')}",
                            "analysis_plan": analysis_plan,
                            "quality_tool_results": [],
                            "quality_plot_paths": [],
                            "quality_narrative": "",
                            "solutions_tool_results": [],
                            "solutions_narrative": "",
                        }

                    tool_results.append({"tool": "infer_schema", "args": fn_args, "result": result_dict})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                    
                except Exception as e:
                    # Catch any MCP client errors
                    error_msg = f"Schema inference tool failed: {str(e)}"
                    analysis_plan = error_msg
                    return {
                        "tool_results": tool_results,
                        "plot_paths": plot_paths,
                        "narrative": error_msg,
                        "analysis_plan": analysis_plan,
                        "quality_tool_results": [],
                        "quality_plot_paths": [],
                        "quality_narrative": "",
                        "solutions_tool_results": [],
                        "solutions_narrative": "",
                    }

            # ----------------------------------------------------------------
            # Phase 2: Planning checkpoint — agent reasons before acting
            # ----------------------------------------------------------------
            messages.append({
                "role": "user",
                "content": (
                    "Based on this schema, reason about which analyses are most appropriate "
                    "for this specific dataset and why. Consider: how many numeric vs categorical "
                    "columns are there? Are there likely correlations? Are there categorical groupings "
                    "worth exploring? Then state your concrete step-by-step plan, naming the exact "
                    "tools and columns you will use."
                ),
            })

            plan_response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=collection_tools,
                tool_choice="none",  # reasoning only — no tool calls
            )
            plan_msg = plan_response.choices[0].message
            analysis_plan = plan_msg.content or ""
            messages.append(plan_msg.model_dump(exclude_unset=True))

            # ----------------------------------------------------------------
            # Phase 3: Execute the plan
            # ----------------------------------------------------------------
            messages.append({"role": "user", "content": "Good. Now execute your plan."})

            for _ in range(max_iterations):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=collection_tools,
                    tool_choice="auto",
                )
                msg = response.choices[0].message
                messages.append(msg.model_dump(exclude_unset=True))

                if not msg.tool_calls:
                    break

                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)

                    mcp_result = await session.call_tool(fn_name, fn_args)

                    result_text = "{}"
                    if mcp_result.content:
                        first = mcp_result.content[0]
                        if hasattr(first, "text") and first.text and first.text.strip():
                            result_text = first.text
                        elif isinstance(first, dict):
                            result_text = first.get("text", "{}")

                    try:
                        result_dict = json.loads(result_text) if result_text.strip() else {}
                    except json.JSONDecodeError:
                        result_dict = {"error": result_text}

                    if "plot_path" in result_dict:
                        plot_paths.append(result_dict["plot_path"])

                    tool_results.append({"tool": fn_name, "args": fn_args, "result": result_dict})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

            # ----------------------------------------------------------------
            # Data Quality Agent  (skipped when "data_quality" not in enabled_categories)
            # ----------------------------------------------------------------
            quality_tool_results: list[dict] = []
            quality_plot_paths: list[str] = []
            quality_narrative: str = ""

            if "data_quality" in _enabled:
                schema_summary = json.dumps(
                    next((r["result"] for r in tool_results if r["tool"] == "infer_schema"), {}),
                    indent=2,
                )

                _quality_image_note = """
IMAGE DATASET NOTE: Columns named feature_0…feature_N are abstract embeddings — non-normality
in these columns is structurally expected, not a data quality issue. Do NOT run `plot_qq` on
feature_N columns. Only run `plot_qq` on human-interpretable metadata columns (width_px,
height_px, file_size_kb, aspect_ratio). Focus quality assessment on coverage gaps, duplicate
images, and metadata consistency.""" if source_type == "image_dataset" else ""

                quality_system_prompt = f"""You are a data quality auditor agent. Your job is to thoroughly
assess the quality of a CSV dataset and identify issues that could affect downstream analysis or
machine learning pipelines. This is directly motivated by research showing that poor data quality
leads to cascading failures in AI systems.

The CSV file path for ALL tool calls is: {csv_path}

You have access to four quality-focused tools:
- `compute_data_quality_score`: Always call this first — it gives you a structured quality breakdown.
- `detect_duplicates`: Always call this to check for duplicate rows.
- `plot_missing_heatmap`: Call this only if the quality score reveals missing values.
- `plot_qq`: Call this for each numeric column to visually assess normality deviations.
{_quality_image_note}
Be thorough. After running all relevant tools, stop and provide a concise quality summary."""

                quality_messages = [
                    {"role": "system", "content": quality_system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Please audit the data quality of this dataset.\n\n"
                            f"Schema (from collection agent):\n{schema_summary}"
                        ),
                    },
                ]

                # Force quality score first
                dq_response = client.chat.completions.create(
                    model=model,
                    messages=quality_messages,
                    tools=openai_tools,
                    tool_choice={"type": "function", "function": {"name": "compute_data_quality_score"}},
                )
                dq_msg = dq_response.choices[0].message
                quality_messages.append(dq_msg.model_dump(exclude_unset=True))

                if dq_msg.tool_calls:
                    tc = dq_msg.tool_calls[0]
                    mcp_result = await session.call_tool("compute_data_quality_score", json.loads(tc.function.arguments))
                    result_text = "{}"
                    if mcp_result.content:
                        first = mcp_result.content[0]
                        if hasattr(first, "text") and first.text:
                            result_text = first.text
                    try:
                        result_dict = json.loads(result_text)
                    except json.JSONDecodeError:
                        result_dict = {"error": result_text}
                    quality_tool_results.append({"tool": "compute_data_quality_score", "args": {}, "result": result_dict})
                    quality_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

                # Let the agent run remaining quality tools autonomously.
                schema_for_cap = next(
                    (r["result"] for r in tool_results if r["tool"] == "infer_schema"), {}
                )
                n_cols = schema_for_cap.get("shape", {}).get("cols", 10)
                quality_iter_cap = max(20, n_cols * 2 + 5)

                for _ in range(quality_iter_cap):
                    dq_response = client.chat.completions.create(
                        model=model,
                        messages=quality_messages,
                        tools=openai_tools,
                        tool_choice="auto",
                    )
                    dq_msg = dq_response.choices[0].message
                    quality_messages.append(dq_msg.model_dump(exclude_unset=True))

                    if not dq_msg.tool_calls:
                        break

                    for tc in dq_msg.tool_calls:
                        fn_name = tc.function.name
                        fn_args = json.loads(tc.function.arguments)
                        mcp_result = await session.call_tool(fn_name, fn_args)

                        result_text = "{}"
                        if mcp_result.content:
                            first = mcp_result.content[0]
                            if hasattr(first, "text") and first.text and first.text.strip():
                                result_text = first.text
                            elif isinstance(first, dict):
                                result_text = first.get("text", "{}")

                        try:
                            result_dict = json.loads(result_text)
                        except json.JSONDecodeError:
                            result_dict = {"error": result_text}

                        if "plot_path" in result_dict:
                            quality_plot_paths.append(result_dict["plot_path"])

                        quality_tool_results.append({"tool": fn_name, "args": fn_args, "result": result_dict})
                        quality_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

                # Quality narrative
                quality_context = json.dumps(
                    [{"tool": r["tool"], "result": r["result"]} for r in quality_tool_results],
                    indent=2,
                )
                quality_report_response = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Based on these data quality audit results, write a concise quality report (2-3 paragraphs). "
                            f"Cover: the overall quality score and what it means, specific issues found and their severity, "
                            f"and concrete recommendations for fixing each issue before using this data in analysis or ML.\n\n"
                            f"Results:\n{quality_context}"
                        ),
                    }],
                )
                quality_narrative = quality_report_response.choices[0].message.content

            # ----------------------------------------------------------------
            # Solutions Agent  (skipped when "solutions" not in enabled_categories)
            # ----------------------------------------------------------------
            solutions_tool_results: list[dict] = []
            solutions_narrative: str = ""

            if "solutions" in _enabled:
                # Collate issues from quality agent results
                detected_issues = []
                for qr in quality_tool_results:
                    if qr["tool"] == "compute_data_quality_score":
                        for issue_dict in qr["result"].get("issues", []):
                            detected_issues.append({
                                "column": issue_dict.get("column", "—"),
                                "issue_type": _map_issue_string_to_type(issue_dict.get("issue", "")),
                                "severity": issue_dict.get("severity", "medium"),
                                "details": {"issue_description": issue_dict.get("issue", "")},
                            })
                    elif qr["tool"] == "detect_anomalies":
                        result = qr["result"]
                        if not result.get("error"):
                            outlier_pct = result.get("outlier_pct", 0)
                            if outlier_pct > 0:
                                detected_issues.append({
                                    "column": result.get("column", "unknown"),
                                    "issue_type": "outliers",
                                    "severity": "high" if outlier_pct > 5 else "medium",
                                    "details": {"outlier_pct": outlier_pct},
                                })
                    elif qr["tool"] == "test_normality":
                        result = qr["result"]
                        if result.get("is_normal") is False:
                            detected_issues.append({
                                "column": result.get("column", "unknown"),
                                "issue_type": "non_normal",
                                "severity": "medium",
                                "details": {"p_value": result.get("p_value", 0.05)},
                            })
                    elif qr["tool"] == "detect_duplicates":
                        result = qr["result"]
                        if result.get("duplicate_rows", 0) > 0:
                            detected_issues.append({
                                "column": "—",
                                "issue_type": "duplicates",
                                "severity": "medium",
                                "details": {"duplicate_pct": result.get("duplicate_pct", 0)},
                            })

                # Also surface high-correlation pairs as multicollinearity issues
                for ar in tool_results:
                    if ar["tool"] == "compute_correlations":
                        for corr_pair in ar["result"].get("top_correlations", []):
                            if abs(corr_pair.get("correlation", 0)) > 0.9:
                                detected_issues.append({
                                    "column": corr_pair.get("col1", "unknown"),
                                    "issue_type": "multicollinearity",
                                    "severity": "medium",
                                    "details": {
                                        "correlation": corr_pair.get("correlation", 0.9),
                                        "correlated_with": corr_pair.get("col2", "unknown"),
                                    },
                                })

                valid_issues = [i for i in detected_issues if isinstance(i, dict) and "issue_type" in i]

                if valid_issues:
                    _modality_note = (
                        "\n\nIMPORTANT CONTEXT: This dataset was derived from images (features extracted via CNN "
                        "or color histograms). Columns named 'feature_0'…'feature_N' are abstract embedding "
                        "dimensions — NOT interpretable data fields. Focus solutions on dataset-level issues "
                        "(duplicates, coverage gaps, class imbalance in image categories) rather than "
                        "per-feature imputation. For outlier/normality issues on feature columns, recommend "
                        "re-running feature extraction or checking for corrupt source images."
                        if source_type == "image_dataset" else ""
                    )
                    sol_response = client.chat.completions.create(
                        model=model,
                        messages=[{
                            "role": "user",
                            "content": (
                                f"You are a data remediation expert. Based on these detected data issues, "
                                f"generate a concise, actionable remediation plan (2-3 paragraphs). For each issue, "
                                f"recommend the best solution, explain why, and provide concrete implementation guidance.\n\n"
                                f"Use this structure:\n"
                                f"1. Prioritize high-severity issues first\n"
                                f"2. For each issue: problem → recommended solution → implementation steps\n"
                                f"3. Suggest a cleaning order (e.g., handle duplicates before imputation)"
                                f"{_modality_note}\n\n"
                                f"Issues detected ({len(valid_issues)} total):\n{json.dumps(valid_issues[:20], indent=2)}"
                            ),
                        }],
                    )
                    solutions_narrative = sol_response.choices[0].message.content

                    issues_to_solve = valid_issues[:100]
                    sol_tool_response = await session.call_tool(
                        "recommend_solutions",
                        {"csv_path": csv_path, "issues": issues_to_solve},
                    )
                    sol_text = "{}"
                    if sol_tool_response.content:
                        first = sol_tool_response.content[0]
                        if hasattr(first, "text") and first.text:
                            sol_text = first.text
                    try:
                        sol_dict = json.loads(sol_text)
                    except json.JSONDecodeError:
                        sol_dict = {"error": "Failed to parse solutions tool response"}
                    solutions_tool_results.append({"tool": "recommend_solutions", "result": sol_dict})

            # ----------------------------------------------------------------
            # Reporting Agent
            # ----------------------------------------------------------------
            context = json.dumps(
                [{"tool": r["tool"], "result": r["result"]} for r in tool_results],
                indent=2,
            )

            _report_modality = (
                "\nNOTE: This dataset was produced by extracting features from images (CNN embeddings or "
                "color histograms). Columns 'feature_0'…'feature_N' represent abstract visual embeddings — "
                "interpret statistical findings in terms of image diversity and visual consistency, not raw "
                "numeric measurements. Metadata columns (filename, width_px, height_px, file_size_kb, etc.) "
                "describe image properties and should be interpreted accordingly."
                if source_type == "image_dataset" else ""
            )
            report_prompt = f"""You are a data science reporting agent. Below are the results of an
automated analysis of a dataset. Write a clear, insightful report (3-5 paragraphs) covering:

1. Dataset overview (size, columns, data types, missing values)
2. Key descriptive statistics and notable distributions
3. Important correlations found (or lack thereof)
4. Anomalies / outliers detected and their potential significance
5. Overall data quality assessment and recommendations
{_report_modality}
Analysis results:
{context}

Write the report in plain English, suitable for a non-technical audience."""

            report_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": report_prompt}],
            )
            narrative = report_response.choices[0].message.content

            return {
                "tool_results": tool_results,
                "plot_paths": plot_paths,
                "narrative": narrative,
                "analysis_plan": analysis_plan,
                "quality_tool_results": quality_tool_results,
                "quality_plot_paths": quality_plot_paths,
                "quality_narrative": quality_narrative,
                "solutions_tool_results": solutions_tool_results,
                "solutions_narrative": solutions_narrative,
            }


def run_pipeline(csv_path: str, model: str = DEFAULT_MODEL, api_key: str = "", enabled_categories: frozenset | None = None) -> dict:
    """
    Synchronous entry point — wraps the async pipeline for Streamlit.

    Automatically detects and processes image datasets, converting them to CSVs
    before running the standard tabular analysis pipeline.
    """
    # Detect image dataset and process if needed
    processed_csv_path, image_metadata = _detect_and_process_images(csv_path, model, api_key)

    # Run the async pipeline on the CSV
    source_type = "image_dataset" if image_metadata else "csv_dataset"
    result = asyncio.run(_run_pipeline_async(processed_csv_path, model, api_key, source_type, enabled_categories))
    
    # Add image metadata to result if applicable
    if image_metadata:
        result["image_processing_metadata"] = image_metadata
        result["source_type"] = "image_dataset"
    else:
        result["source_type"] = "csv_dataset"
    
    return result


# ---------------------------------------------------------------------------
# Two-phase API: plan-only + execution (used by Streamlit for human-in-loop)
# ---------------------------------------------------------------------------

def _extract_planned_tools(plan_text: str, available_tools: list) -> list:
    """Return tool names mentioned in the plan text (keyword match)."""
    plan_lower = plan_text.lower()
    return [t for t in available_tools if t in plan_lower or t.replace("_", " ") in plan_lower]


async def _run_plan_phase_async(
    csv_path: str,
    model: str,
    api_key: str = "",
    source_type: str = "csv_dataset",
    enabled_categories: frozenset | None = None,
    on_event=None,
) -> dict:
    """Run schema inference + planning only. Returns state dict for user review."""
    def emit(event: dict):
        if on_event:
            on_event(event)

    server_params = StdioServerParameters(command=sys.executable, args=[MCP_SERVER_PATH], env=None)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            emit({"type": "phase_start", "phase": "mcp_init"})

            mcp_tools_response = await session.list_tools()
            mcp_tools = mcp_tools_response.tools
            openai_tools = [_mcp_tool_to_openai(t) for t in mcp_tools]
            tool_descriptions = {t.name: t.description for t in mcp_tools}

            _enabled = enabled_categories if enabled_categories is not None else _ALL_CATEGORIES
            _collection_allowed = set(_ALWAYS_TOOLS)
            if "visualizations" in _enabled:
                _collection_allowed |= _VIZ_TOOLS
            if "anomaly_detection" in _enabled:
                _collection_allowed |= _ANOMALY_TOOLS
            collection_tools = [t for t in openai_tools if t["function"]["name"] in _collection_allowed]
            collection_tool_names = [t["function"]["name"] for t in collection_tools]

            effective_key = api_key or OPENAI_API_KEY
            client = OpenAI(api_key=effective_key, base_url=OPENAI_BASE_URL)

            _image_rules = """
IMAGE DATASET RULES (this CSV was derived from image feature extraction):
- Columns named feature_0…feature_N are abstract CNN/histogram embeddings — NOT interpretable individually.
- Do NOT run `plot_distribution` or `detect_anomalies` or `test_normality` on feature_N columns — they produce noise, not insight.
- DO run `summarize_statistics`, `compute_correlations`, `plot_correlation_heatmap`, and `plot_pairplot` on ALL columns.
- DO run `plot_distribution`, `detect_anomalies`, and `test_normality` on metadata columns only (filename, width_px, height_px, file_size_kb, aspect_ratio).
- For `plot_boxplot`: use a metadata numeric column grouped by any categorical column if present.
- Interpret findings in terms of image diversity, coverage gaps, and metadata anomalies.""" if source_type == "image_dataset" else ""

            system_prompt = f"""You are an expert data analyst agent. You have access to tools that
operate on CSV files. Your goal is to produce the most insightful analysis possible for the
specific dataset you are given — not to blindly run every tool.

The CSV file path for ALL tool calls is: {csv_path}

DECISION RULES — apply these based on what you observe in the data:
- Run `summarize_statistics` and `test_normality` for every numeric column found.
- Run `compute_correlations`, `plot_correlation_heatmap`, and `plot_pairplot` ONLY if there are 2+ numeric columns.
- Run `plot_scatter` ONLY for column pairs where |correlation| > 0.5.
- Run `plot_boxplot` ONLY if there are both numeric columns and categorical columns with ≤15 unique values.
- Run `plot_distribution` for every column (numeric or categorical).
- Run `detect_anomalies` for every numeric column.
- Do NOT call a tool that is irrelevant to this dataset's structure.

VISUALIZATION BUDGET — generate at most 6 plots total. Quality over quantity.
{_image_rules}
You will first be asked to reason and plan. Then execute your plan."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Please analyse this CSV file: {csv_path}"},
            ]
            tool_results: list[dict] = []

            # Schema phase
            emit({"type": "phase_start", "phase": "schema"})
            schema_response = client.chat.completions.create(
                model=model, messages=messages, tools=collection_tools,
                tool_choice={"type": "function", "function": {"name": "infer_schema"}},
            )
            schema_msg = schema_response.choices[0].message
            messages.append(schema_msg.model_dump(exclude_unset=True))

            if schema_msg.tool_calls:
                tc = schema_msg.tool_calls[0]
                fn_args = json.loads(tc.function.arguments)
                emit({"type": "tool_start", "tool": "infer_schema", "args": fn_args})
                try:
                    mcp_result = await session.call_tool("infer_schema", fn_args)
                    result_text = "{}"
                    if mcp_result.content:
                        first = mcp_result.content[0]
                        if hasattr(first, "text") and first.text:
                            result_text = first.text
                    try:
                        result_dict = json.loads(result_text) if result_text.strip() else {}
                    except json.JSONDecodeError:
                        result_dict = {"error": result_text}
                    tool_results.append({"tool": "infer_schema", "args": fn_args, "result": result_dict})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                    emit({"type": "tool_result", "tool": "infer_schema", "result": result_dict})
                except Exception as e:
                    result_dict = {"error": str(e)}
                    tool_results.append({"tool": "infer_schema", "args": fn_args, "result": result_dict})

            # Planning phase
            emit({"type": "phase_start", "phase": "planning"})
            messages.append({
                "role": "user",
                "content": (
                    "Based on this schema, reason about which analyses are most appropriate "
                    "for this specific dataset and why. Consider: how many numeric vs categorical "
                    "columns are there? Are there likely correlations? Are there categorical groupings "
                    "worth exploring? Then state your concrete step-by-step plan, naming the exact "
                    "tools and columns you will use."
                ),
            })
            plan_response = client.chat.completions.create(
                model=model, messages=messages, tools=collection_tools, tool_choice="none",
            )
            plan_msg = plan_response.choices[0].message
            analysis_plan = plan_msg.content or ""
            messages.append(plan_msg.model_dump(exclude_unset=True))
            emit({"type": "plan_ready", "text": analysis_plan})

            planned_tools = _extract_planned_tools(analysis_plan, collection_tool_names)
            schema_result = tool_results[0]["result"] if tool_results else {}

            return {
                "analysis_plan": analysis_plan,
                "messages": messages,
                "available_tool_names": collection_tool_names,
                "planned_tool_names": planned_tools,
                "schema_result": schema_result,
                "tool_descriptions": tool_descriptions,
                "enabled_categories": _enabled,
                "csv_path": csv_path,
                "source_type": source_type,
            }


async def _run_execution_async(
    plan_state: dict,
    approved_tools: list,
    model: str,
    api_key: str = "",
    on_event=None,
) -> dict:
    """Run collection + quality + solutions + reporting given a plan state and approved tools."""
    def emit(event: dict):
        if on_event:
            on_event(event)

    csv_path = plan_state["csv_path"]
    source_type = plan_state.get("source_type", "csv_dataset")
    _enabled = plan_state.get("enabled_categories", _ALL_CATEGORIES)

    outputs_dir = os.path.join(os.path.dirname(MCP_SERVER_PATH), "outputs")
    for old_file in glob.glob(os.path.join(outputs_dir, "*.png")):
        try:
            os.remove(old_file)
        except OSError:
            pass

    server_params = StdioServerParameters(command=sys.executable, args=[MCP_SERVER_PATH], env=None)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools_response = await session.list_tools()
            mcp_tools = mcp_tools_response.tools
            all_openai_tools = [_mcp_tool_to_openai(t) for t in mcp_tools]
            collection_tools = [t for t in all_openai_tools if t["function"]["name"] in approved_tools]

            effective_key = api_key or OPENAI_API_KEY
            client = OpenAI(api_key=effective_key, base_url=OPENAI_BASE_URL)

            messages = list(plan_state["messages"])
            analysis_plan = plan_state["analysis_plan"]
            schema_result = plan_state.get("schema_result", {})

            # Seed tool_results with schema from plan phase
            tool_results: list[dict] = [{"tool": "infer_schema", "args": {}, "result": schema_result}]
            plot_paths: list[str] = []

            approved_str = ", ".join(approved_tools) if approved_tools else "none"
            messages.append({
                "role": "user",
                "content": f"The user approved these tools: {approved_str}. Now execute your plan using only these approved tools.",
            })

            # Collection phase
            emit({"type": "phase_start", "phase": "collection"})
            for _ in range(40):
                response = client.chat.completions.create(
                    model=model, messages=messages,
                    tools=collection_tools if collection_tools else all_openai_tools,
                    tool_choice="auto",
                )
                msg = response.choices[0].message
                messages.append(msg.model_dump(exclude_unset=True))
                if not msg.tool_calls:
                    break
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)
                    emit({"type": "tool_start", "tool": fn_name, "args": fn_args})
                    mcp_result = await session.call_tool(fn_name, fn_args)
                    result_text = "{}"
                    if mcp_result.content:
                        first = mcp_result.content[0]
                        if hasattr(first, "text") and first.text and first.text.strip():
                            result_text = first.text
                        elif isinstance(first, dict):
                            result_text = first.get("text", "{}")
                    try:
                        result_dict = json.loads(result_text) if result_text.strip() else {}
                    except json.JSONDecodeError:
                        result_dict = {"error": result_text}
                    if "plot_path" in result_dict:
                        plot_paths.append(result_dict["plot_path"])
                    tool_results.append({"tool": fn_name, "args": fn_args, "result": result_dict})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                    emit({"type": "tool_result", "tool": fn_name, "result": result_dict})

            # Quality agent
            quality_tool_results: list[dict] = []
            quality_plot_paths: list[str] = []
            quality_narrative: str = ""

            if "data_quality" in _enabled:
                schema_summary = json.dumps(schema_result, indent=2)
                _quality_image_note = """
IMAGE DATASET NOTE: Columns named feature_0…feature_N are abstract embeddings — non-normality
in these columns is structurally expected, not a data quality issue. Do NOT run `plot_qq` on
feature_N columns. Only run `plot_qq` on human-interpretable metadata columns.""" if source_type == "image_dataset" else ""

                quality_system_prompt = f"""You are a data quality auditor agent. Your job is to thoroughly
assess the quality of a CSV dataset and identify issues that could affect downstream analysis or
machine learning pipelines.

The CSV file path for ALL tool calls is: {csv_path}

You have access to four quality-focused tools:
- `compute_data_quality_score`: Always call this first — it gives you a structured quality breakdown.
- `detect_duplicates`: Always call this to check for duplicate rows.
- `plot_missing_heatmap`: Call this only if the quality score reveals missing values.
- `plot_qq`: Call this for each numeric column to visually assess normality deviations.
{_quality_image_note}
Be thorough. After running all relevant tools, stop and provide a concise quality summary."""

                quality_messages = [
                    {"role": "system", "content": quality_system_prompt},
                    {"role": "user", "content": f"Please audit the data quality of this dataset.\n\nSchema:\n{schema_summary}"},
                ]

                emit({"type": "phase_start", "phase": "quality_score"})
                dq_response = client.chat.completions.create(
                    model=model, messages=quality_messages, tools=all_openai_tools,
                    tool_choice={"type": "function", "function": {"name": "compute_data_quality_score"}},
                )
                dq_msg = dq_response.choices[0].message
                quality_messages.append(dq_msg.model_dump(exclude_unset=True))
                if dq_msg.tool_calls:
                    tc = dq_msg.tool_calls[0]
                    fn_args = json.loads(tc.function.arguments)
                    emit({"type": "tool_start", "tool": "compute_data_quality_score", "args": fn_args})
                    mcp_result = await session.call_tool("compute_data_quality_score", fn_args)
                    result_text = "{}"
                    if mcp_result.content:
                        first = mcp_result.content[0]
                        if hasattr(first, "text") and first.text:
                            result_text = first.text
                    try:
                        result_dict = json.loads(result_text)
                    except json.JSONDecodeError:
                        result_dict = {"error": result_text}
                    quality_tool_results.append({"tool": "compute_data_quality_score", "args": fn_args, "result": result_dict})
                    quality_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                    emit({"type": "tool_result", "tool": "compute_data_quality_score", "result": result_dict})

                n_cols = schema_result.get("shape", {}).get("cols", 10)
                quality_iter_cap = max(20, n_cols * 2 + 5)
                emit({"type": "phase_start", "phase": "quality_tools"})
                for _ in range(quality_iter_cap):
                    dq_response = client.chat.completions.create(
                        model=model, messages=quality_messages, tools=all_openai_tools, tool_choice="auto",
                    )
                    dq_msg = dq_response.choices[0].message
                    quality_messages.append(dq_msg.model_dump(exclude_unset=True))
                    if not dq_msg.tool_calls:
                        break
                    for tc in dq_msg.tool_calls:
                        fn_name = tc.function.name
                        fn_args = json.loads(tc.function.arguments)
                        emit({"type": "tool_start", "tool": fn_name, "args": fn_args})
                        mcp_result = await session.call_tool(fn_name, fn_args)
                        result_text = "{}"
                        if mcp_result.content:
                            first = mcp_result.content[0]
                            if hasattr(first, "text") and first.text and first.text.strip():
                                result_text = first.text
                            elif isinstance(first, dict):
                                result_text = first.get("text", "{}")
                        try:
                            result_dict = json.loads(result_text)
                        except json.JSONDecodeError:
                            result_dict = {"error": result_text}
                        if "plot_path" in result_dict:
                            quality_plot_paths.append(result_dict["plot_path"])
                        quality_tool_results.append({"tool": fn_name, "args": fn_args, "result": result_dict})
                        quality_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})
                        emit({"type": "tool_result", "tool": fn_name, "result": result_dict})

                emit({"type": "phase_start", "phase": "quality_narrative"})
                quality_context = json.dumps(
                    [{"tool": r["tool"], "result": r["result"]} for r in quality_tool_results], indent=2,
                )
                quality_report_response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": (
                        f"Based on these data quality audit results, write a concise quality report (2-3 paragraphs). "
                        f"Cover: the overall quality score and what it means, specific issues found and their severity, "
                        f"and concrete recommendations for fixing each issue.\n\nResults:\n{quality_context}"
                    )}],
                )
                quality_narrative = quality_report_response.choices[0].message.content

            # Solutions agent
            solutions_tool_results: list[dict] = []
            solutions_narrative: str = ""

            if "solutions" in _enabled:
                detected_issues = []
                for qr in quality_tool_results:
                    if qr["tool"] == "compute_data_quality_score":
                        for issue_dict in qr["result"].get("issues", []):
                            detected_issues.append({
                                "column": issue_dict.get("column", "—"),
                                "issue_type": _map_issue_string_to_type(issue_dict.get("issue", "")),
                                "severity": issue_dict.get("severity", "medium"),
                                "details": {"issue_description": issue_dict.get("issue", "")},
                            })
                    elif qr["tool"] == "detect_anomalies":
                        result = qr["result"]
                        if not result.get("error") and result.get("outlier_pct", 0) > 0:
                            detected_issues.append({
                                "column": result.get("column", "unknown"),
                                "issue_type": "outliers",
                                "severity": "high" if result.get("outlier_pct", 0) > 5 else "medium",
                                "details": {"outlier_pct": result.get("outlier_pct", 0)},
                            })
                    elif qr["tool"] == "test_normality":
                        result = qr["result"]
                        if result.get("is_normal") is False:
                            detected_issues.append({
                                "column": result.get("column", "unknown"),
                                "issue_type": "non_normal",
                                "severity": "medium",
                                "details": {"p_value": result.get("p_value", 0.05)},
                            })
                    elif qr["tool"] == "detect_duplicates":
                        result = qr["result"]
                        if result.get("duplicate_rows", 0) > 0:
                            detected_issues.append({
                                "column": "—",
                                "issue_type": "duplicates",
                                "severity": "medium",
                                "details": {"duplicate_pct": result.get("duplicate_pct", 0)},
                            })
                for ar in tool_results:
                    if ar["tool"] == "compute_correlations":
                        for corr_pair in ar["result"].get("top_correlations", []):
                            if abs(corr_pair.get("correlation", 0)) > 0.9:
                                detected_issues.append({
                                    "column": corr_pair.get("col1", "unknown"),
                                    "issue_type": "multicollinearity",
                                    "severity": "medium",
                                    "details": {"correlation": corr_pair.get("correlation", 0.9), "correlated_with": corr_pair.get("col2", "unknown")},
                                })

                valid_issues = [i for i in detected_issues if isinstance(i, dict) and "issue_type" in i]
                if valid_issues:
                    emit({"type": "phase_start", "phase": "solutions"})
                    _modality_note = (
                        "\n\nIMPORTANT: This dataset was derived from images. Focus solutions on dataset-level issues "
                        "(duplicates, coverage gaps, class imbalance) rather than per-feature imputation."
                        if source_type == "image_dataset" else ""
                    )
                    sol_response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": (
                            f"You are a data remediation expert. Based on these detected issues, generate a concise, "
                            f"actionable remediation plan (2-3 paragraphs). Prioritize high-severity issues first, "
                            f"and suggest a cleaning order.{_modality_note}\n\n"
                            f"Issues ({len(valid_issues)} total):\n{json.dumps(valid_issues[:20], indent=2)}"
                        )}],
                    )
                    solutions_narrative = sol_response.choices[0].message.content
                    sol_tool_response = await session.call_tool(
                        "recommend_solutions", {"csv_path": csv_path, "issues": valid_issues[:100]},
                    )
                    sol_text = "{}"
                    if sol_tool_response.content:
                        first = sol_tool_response.content[0]
                        if hasattr(first, "text") and first.text:
                            sol_text = first.text
                    try:
                        sol_dict = json.loads(sol_text)
                    except json.JSONDecodeError:
                        sol_dict = {"error": "Failed to parse solutions response"}
                    solutions_tool_results.append({"tool": "recommend_solutions", "result": sol_dict})
                    emit({"type": "tool_result", "tool": "recommend_solutions", "result": sol_dict})

            # Reporting agent
            emit({"type": "phase_start", "phase": "reporting"})
            context = json.dumps(
                [{"tool": r["tool"], "result": r["result"]} for r in tool_results], indent=2,
            )
            _report_modality = (
                "\nNOTE: This dataset was produced by extracting features from images. "
                "Interpret findings in terms of image diversity and visual consistency."
                if source_type == "image_dataset" else ""
            )
            report_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": (
                    f"You are a data science reporting agent. Write a clear, insightful report (3-5 paragraphs) covering: "
                    f"1. Dataset overview, 2. Key statistics and distributions, 3. Important correlations, "
                    f"4. Anomalies detected, 5. Data quality assessment and recommendations."
                    f"{_report_modality}\n\nAnalysis results:\n{context}\n\n"
                    f"Write in plain English for a non-technical audience."
                )}],
            )
            narrative = report_response.choices[0].message.content

            return {
                "tool_results": tool_results,
                "plot_paths": plot_paths,
                "narrative": narrative,
                "analysis_plan": analysis_plan,
                "quality_tool_results": quality_tool_results,
                "quality_plot_paths": quality_plot_paths,
                "quality_narrative": quality_narrative,
                "solutions_tool_results": solutions_tool_results,
                "solutions_narrative": solutions_narrative,
                "source_type": source_type,
            }


def run_plan_only(
    csv_path: str,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    enabled_categories: frozenset | None = None,
    on_event=None,
) -> dict:
    """Run schema + planning phase only. Returns plan_state dict for user review."""
    processed_csv_path, image_metadata = _detect_and_process_images(csv_path, model, api_key)
    source_type = "image_dataset" if image_metadata else "csv_dataset"
    plan_state = asyncio.run(_run_plan_phase_async(
        processed_csv_path, model, api_key, source_type, enabled_categories, on_event,
    ))
    if image_metadata:
        plan_state["image_processing_metadata"] = image_metadata
    return plan_state


def run_execution_phase(
    plan_state: dict,
    approved_tools: list,
    model: str = DEFAULT_MODEL,
    api_key: str = "",
    on_event=None,
) -> dict:
    """Run collection + quality + solutions + reporting with user-approved tools."""
    result = asyncio.run(_run_execution_async(plan_state, approved_tools, model, api_key, on_event))
    if plan_state.get("image_processing_metadata"):
        result["image_processing_metadata"] = plan_state["image_processing_metadata"]
    return result


# ---------------------------------------------------------------------------
# CLI entry point for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py <path_to_csv>")
        sys.exit(1)

    csv_path = os.path.abspath(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    print(f"Running pipeline on: {csv_path} with model: {model}\n")

    results = run_pipeline(csv_path, model)

    print("=" * 60)
    print("ANALYSIS PLAN:")
    print(results["analysis_plan"])

    print("\nTOOL CALLS EXECUTED:")
    for r in results["tool_results"]:
        print(f"  - {r['tool']}({r['args']})")

    print("\nPLOTS GENERATED:")
    for p in results["plot_paths"]:
        print(f"  - {p}")

    print("\nNARRATIVE REPORT:")
    print(results["narrative"])
