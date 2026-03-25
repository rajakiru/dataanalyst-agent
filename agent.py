"""
Agent orchestration: collection agent + reporting agent.
Uses Dedalus (OpenAI-compatible) for LLM calls and MCP stdio for tool execution.
"""

import asyncio
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

load_dotenv()

DEDALUS_BASE_URL = os.getenv("DEDALUS_BASE_URL", "https://api.dedaluslabs.ai")
DEDALUS_API_KEY = os.getenv("DEDALUS_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

MCP_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server.py")


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


async def _run_pipeline_async(csv_path: str, model: str) -> dict:
    """
    Full async pipeline:
    1. Spawn MCP server subprocess (stdio).
    2. Collection agent: LLM decides which tools to call, executes them.
    3. Reporting agent: LLM summarises all tool outputs into a narrative.
    Returns a dict with all results.
    """
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

            # ----------------------------------------------------------------
            # Collection Agent
            # ----------------------------------------------------------------
            client = OpenAI(api_key=DEDALUS_API_KEY, base_url=DEDALUS_BASE_URL)

            system_prompt = f"""You are an expert data analyst agent. You have access to tools that
operate on CSV files. Your goal is to produce the most insightful analysis possible for the
specific dataset you are given — not to blindly run every tool.

The CSV file path for ALL tool calls is: {csv_path}

DECISION RULES — apply these based on what you observe in the data:
- Run `summarize_statistics` and `test_normality` for every numeric column found.
- Run `compute_correlations`, `plot_correlation_heatmap`, and `plot_pairplot` ONLY if there are 2+ numeric columns.
- Run `plot_scatter` ONLY for column pairs where |correlation| > 0.5. Skip it if no strong correlations exist.
- Run `plot_boxplot` ONLY if there are both numeric columns and categorical columns with ≤15 unique values.
- Run `plot_distribution` for every column (numeric or categorical).
- Run `detect_anomalies` for every numeric column.
- If a column has high skewness (>1 or <-1), note this when interpreting its anomaly results.
- Do NOT call a tool that is irrelevant to this dataset's structure.

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
                tools=openai_tools,
                tool_choice={"type": "function", "function": {"name": "infer_schema"}},
            )
            schema_msg = schema_response.choices[0].message
            messages.append(schema_msg.model_dump(exclude_unset=True))

            # Execute the forced infer_schema call
            if schema_msg.tool_calls:
                tc = schema_msg.tool_calls[0]
                fn_args = json.loads(tc.function.arguments)
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
                    tools=openai_tools,
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
            # Reporting Agent
            # ----------------------------------------------------------------
            context = json.dumps(
                [{"tool": r["tool"], "result": r["result"]} for r in tool_results],
                indent=2,
            )

            report_prompt = f"""You are a data science reporting agent. Below are the results of an
automated analysis of a CSV dataset. Write a clear, insightful report (3-5 paragraphs) covering:

1. Dataset overview (size, columns, data types, missing values)
2. Key descriptive statistics and notable distributions
3. Important correlations found (or lack thereof)
4. Anomalies / outliers detected and their potential significance
5. Overall data quality assessment and recommendations

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
            }


def run_pipeline(csv_path: str, model: str = DEFAULT_MODEL) -> dict:
    """Synchronous entry point — wraps the async pipeline for Streamlit."""
    return asyncio.run(_run_pipeline_async(csv_path, model))


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
