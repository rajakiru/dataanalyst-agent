"use client";

import { useState } from "react";
import { AppState, Step } from "../WorkflowBuilder";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";

const ALWAYS_ON = [
  "compute_data_quality_score",
  "detect_duplicates",
  "plot_missing_heatmap",
  "plot_qq",
];

type Props = {
  appState: AppState;
  setAppState: React.Dispatch<React.SetStateAction<AppState>>;
  setActiveStep: (id: Step["id"]) => void;
  markDone: (id: Step["id"]) => void;
};

export function ReviewStep({ appState, setAppState, setActiveStep, markDone }: Props) {
  const planState = appState.planState as {
    analysis_plan?: string;
    planned_tool_names?: string[];
    available_tool_names?: string[];
    tool_descriptions?: Record<string, string>;
  } | null;

  const allTools: string[] = planState?.available_tool_names ?? [];
  const plannedTools: string[] = planState?.planned_tool_names ?? allTools;
  const descriptions: Record<string, string> = planState?.tool_descriptions ?? {};

  const [checked, setChecked] = useState<Record<string, boolean>>(
    Object.fromEntries(allTools.map((t) => [t, plannedTools.includes(t) || ALWAYS_ON.includes(t)]))
  );
  const [loading, setLoading] = useState(false);

  function toggle(tool: string) {
    if (ALWAYS_ON.includes(tool)) return;
    setChecked((p) => ({ ...p, [tool]: !p[tool] }));
  }

  async function handleRun() {
    const approvedTools = Object.entries(checked)
      .filter(([, v]) => v)
      .map(([k]) => k);

    setAppState((p) => ({ ...p, approvedTools }));
    setLoading(true);
    markDone("review");
    setActiveStep("collection");

    try {
      const res = await fetch("/api/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan_state: appState.planState, approved_tools: approvedTools }),
        signal: AbortSignal.timeout(290_000),
      });
      const results = await res.json();
      setAppState((p) => ({ ...p, results }));
    } catch (e) {
      alert(`Execution failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-2xl mx-auto py-16 px-8">
      <h2 className="text-xl font-semibold text-zinc-900 mb-1">Review Plan</h2>
      <p className="text-sm text-zinc-500 mb-6">
        The agent has drafted an analysis plan. Select which tools to include.
      </p>

      {/* Analysis plan */}
      {planState?.analysis_plan && (
        <div className="mb-8 rounded-xl border border-zinc-200 bg-white p-5">
          <p className="text-xs font-medium text-zinc-400 uppercase tracking-widest mb-3">
            Agent plan
          </p>
          <p className="text-sm text-zinc-700 whitespace-pre-wrap leading-relaxed">
            {planState.analysis_plan}
          </p>
        </div>
      )}

      {/* Tool checklist */}
      <div className="rounded-xl border border-zinc-200 bg-white divide-y divide-zinc-100">
        <div className="px-5 py-3">
          <p className="text-xs font-medium text-zinc-400 uppercase tracking-widest">Tools</p>
        </div>
        {allTools.length === 0 && (
          <div className="px-5 py-4 text-sm text-zinc-400">No tools available</div>
        )}
        {allTools.map((tool) => {
          const alwaysOn = ALWAYS_ON.includes(tool);
          return (
            <div
              key={tool}
              className={`flex items-start gap-3 px-5 py-3.5 ${alwaysOn ? "opacity-60" : ""}`}
            >
              <Checkbox
                id={tool}
                checked={checked[tool] ?? false}
                onCheckedChange={() => toggle(tool)}
                disabled={alwaysOn}
                className="mt-0.5"
              />
              <div className="flex-1 min-w-0">
                <label
                  htmlFor={tool}
                  className="text-sm font-medium text-zinc-800 cursor-pointer font-mono"
                >
                  {tool}
                </label>
                {descriptions[tool] && (
                  <p className="text-xs text-zinc-400 mt-0.5 leading-relaxed">
                    {descriptions[tool]}
                  </p>
                )}
              </div>
              {alwaysOn && (
                <Badge variant="secondary" className="text-xs shrink-0">
                  always on
                </Badge>
              )}
              {plannedTools.includes(tool) && !alwaysOn && (
                <Badge variant="outline" className="text-xs shrink-0 text-indigo-600 border-indigo-200">
                  planned
                </Badge>
              )}
            </div>
          );
        })}
      </div>

      <button
        onClick={handleRun}
        disabled={loading}
        className="mt-8 w-full py-3 rounded-xl bg-indigo-600 text-white font-medium text-sm hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? "Running pipeline…" : "Run Analysis →"}
      </button>
    </div>
  );
}
