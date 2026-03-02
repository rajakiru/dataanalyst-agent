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

            system_prompt = f"""You are an expert data analyst agent. You have access to a set of
analysis tools that operate on CSV files. Your job is to thoroughly analyse the provided dataset.

The CSV file path you must use for ALL tool calls is:
  {csv_path}

Always pass this exact path as the `csv_path` argument to every tool.

Follow this strategy:
1. Call `infer_schema` with csv_path="{csv_path}" to understand the dataset structure.
2. Call `summarize_statistics` with csv_path="{csv_path}" to get descriptive stats.
3. If there are 2+ numeric columns, call `compute_correlations` and `plot_correlation_heatmap`.
4. For each numeric column identified in the schema, call `plot_distribution`.
5. For each numeric column, call `detect_anomalies`.
6. Once you have collected all results, stop calling tools and provide a brief completion message.

Be systematic and thorough. Do not skip columns."""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Please analyse this CSV file: {csv_path}",
                },
            ]

            tool_results: list[dict] = []
            plot_paths: list[str] = []
            max_iterations = 30  # safety cap

            for _ in range(max_iterations):
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                )
                msg = response.choices[0].message

                # Append assistant message to history
                messages.append(msg.model_dump(exclude_unset=True))

                # If no tool calls, the agent is done
                if not msg.tool_calls:
                    break

                # Execute each tool call via MCP
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments)

                    mcp_result = await session.call_tool(fn_name, fn_args)

                    # Extract text from the first content block safely
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

                    # Collect plot paths separately
                    if "plot_path" in result_dict:
                        plot_paths.append(result_dict["plot_path"])

                    tool_results.append({
                        "tool": fn_name,
                        "args": fn_args,
                        "result": result_dict,
                    })

                    # Append tool result to message history
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })

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
    print("TOOL CALLS EXECUTED:")
    for r in results["tool_results"]:
        print(f"  - {r['tool']}({r['args']})")

    print("\nPLOTS GENERATED:")
    for p in results["plot_paths"]:
        print(f"  - {p}")

    print("\nNARRATIVE REPORT:")
    print(results["narrative"])
