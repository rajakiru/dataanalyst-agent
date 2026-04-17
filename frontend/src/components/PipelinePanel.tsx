"use client";

import { Step } from "./WorkflowBuilder";
import { CheckCircle2, Circle, Loader2 } from "lucide-react";

const STEP_ICONS: Record<Step["id"], string> = {
  upload: "📁",
  review: "🔍",
  collection: "📊",
  diagnosis: "🩺",
  intervention: "🔧",
  report: "📄",
};

type Props = {
  steps: Step[];
  activeStep: Step["id"];
  onSelect: (id: Step["id"]) => void;
};

export function PipelinePanel({ steps, activeStep, onSelect }: Props) {
  return (
    <div className="flex flex-col py-6 px-4 gap-1">
      <p className="text-xs font-medium text-zinc-400 uppercase tracking-widest px-2 mb-3">
        Pipeline
      </p>
      {steps.map((step, i) => {
        const isActive = step.id === activeStep;
        const isDone = step.status === "done";
        const isRunning = step.status === "active" && !isActive;

        return (
          <div key={step.id} className="flex flex-col">
            <button
              onClick={() => onSelect(step.id)}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors text-left w-full ${
                isActive
                  ? "bg-indigo-50 text-indigo-700 font-medium"
                  : "text-zinc-600 hover:bg-zinc-50"
              }`}
            >
              <span className="text-base shrink-0">{STEP_ICONS[step.id]}</span>
              <span className="flex-1">{step.label}</span>
              {isDone && <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />}
              {isRunning && <Loader2 className="w-4 h-4 text-indigo-400 animate-spin shrink-0" />}
              {!isDone && !isRunning && isActive && (
                <Circle className="w-4 h-4 text-indigo-400 shrink-0" />
              )}
            </button>

            {/* Connector line */}
            {i < steps.length - 1 && (
              <div className="ml-[22px] w-px h-4 bg-zinc-200" />
            )}
          </div>
        );
      })}
    </div>
  );
}
