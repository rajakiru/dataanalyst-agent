"use client";

import { useState, useRef } from "react";
import { PipelinePanel } from "./PipelinePanel";
import { ConfigPanel } from "./ConfigPanel";
import { AssistantChat } from "./AssistantChat";
import { ResultsTabs } from "./ResultsTabs";

export type Step =
  | { id: "upload"; label: "Upload Dataset"; status: "idle" | "active" | "done" }
  | { id: "review"; label: "Review Plan"; status: "idle" | "active" | "done" }
  | { id: "collection"; label: "Collection Agent"; status: "idle" | "active" | "done" }
  | { id: "diagnosis"; label: "Diagnosis Agent"; status: "idle" | "active" | "done" }
  | { id: "intervention"; label: "Intervention Agent"; status: "idle" | "active" | "done" }
  | { id: "report"; label: "Report"; status: "idle" | "active" | "done" };

export type AppState = {
  activeStep: Step["id"];
  file: File | null;
  sampleDataset: string | null;
  planState: Record<string, unknown> | null;
  approvedTools: string[];
  results: Record<string, unknown> | null;
  assistantOpen: boolean;
};

const INITIAL_STEPS: Step[] = [
  { id: "upload", label: "Upload Dataset", status: "idle" },
  { id: "review", label: "Review Plan", status: "idle" },
  { id: "collection", label: "Collection Agent", status: "idle" },
  { id: "diagnosis", label: "Diagnosis Agent", status: "idle" },
  { id: "intervention", label: "Intervention Agent", status: "idle" },
  { id: "report", label: "Report", status: "idle" },
];

export function WorkflowBuilder() {
  const [steps, setSteps] = useState<Step[]>(INITIAL_STEPS);
  const [appState, setAppState] = useState<AppState>({
    activeStep: "upload",
    file: null,
    sampleDataset: null,
    planState: null,
    approvedTools: [],
    results: null,
    assistantOpen: false,
  });

  function setActiveStep(id: Step["id"]) {
    setSteps((prev) =>
      prev.map((s) =>
        s.id === id
          ? { ...s, status: "active" }
          : s.status === "active"
          ? { ...s, status: "idle" }
          : s
      )
    );
    setAppState((prev) => ({ ...prev, activeStep: id }));
  }

  function markDone(id: Step["id"]) {
    setSteps((prev) =>
      prev.map((s) => (s.id === id ? { ...s, status: "done" } : s))
    );
  }

  function reset() {
    setSteps(INITIAL_STEPS);
    setAppState({
      activeStep: "upload",
      file: null,
      sampleDataset: null,
      planState: null,
      approvedTools: [],
      results: null,
      assistantOpen: false,
    });
  }

  return (
    <div className="flex h-screen w-screen bg-[#f5f6fa] font-sans overflow-hidden">
      {/* Top bar */}
      <div className="absolute top-0 left-0 right-0 h-12 bg-white border-b border-zinc-200 flex items-center px-6 gap-4 z-10">
        <span className="font-semibold text-zinc-900 text-sm tracking-tight">AutoAnalyst</span>
        <span className="text-zinc-300">|</span>
        <span className="text-zinc-500 text-sm">Multi-Agent Data Cascade Debugger</span>
        <div className="ml-auto flex gap-2">
          <button
            onClick={() => setAppState((p) => ({ ...p, assistantOpen: !p.assistantOpen }))}
            className="text-xs px-3 py-1.5 rounded-md bg-indigo-600 text-white hover:bg-indigo-700 transition-colors"
          >
            Assistant
          </button>
          <button
            onClick={reset}
            className="text-xs px-3 py-1.5 rounded-md border border-zinc-200 text-zinc-600 hover:bg-zinc-50 transition-colors"
          >
            ← New analysis
          </button>
        </div>
      </div>

      {/* Main layout */}
      <div className="flex w-full h-full pt-12">
        {/* Left: pipeline steps */}
        <div className="w-72 shrink-0 border-r border-zinc-200 bg-white overflow-y-auto">
          <PipelinePanel
            steps={steps}
            activeStep={appState.activeStep}
            onSelect={setActiveStep}
          />
        </div>

        {/* Center: config / results */}
        <div className="flex-1 overflow-y-auto">
          {appState.results ? (
            <ResultsTabs results={appState.results} onReset={reset} />
          ) : (
            <ConfigPanel
              appState={appState}
              setAppState={setAppState}
              setActiveStep={setActiveStep}
              markDone={markDone}
            />
          )}
        </div>

        {/* Right: assistant chat (slide-in) */}
        {appState.assistantOpen && (
          <div className="w-80 shrink-0 border-l border-zinc-200 bg-white">
            <AssistantChat onClose={() => setAppState((p) => ({ ...p, assistantOpen: false }))} />
          </div>
        )}
      </div>
    </div>
  );
}
