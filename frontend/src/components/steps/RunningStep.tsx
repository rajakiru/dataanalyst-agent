"use client";

import { AppState, Step } from "../WorkflowBuilder";
import { Loader2 } from "lucide-react";

const PHASE_DESCRIPTIONS: Record<string, string> = {
  collection: "Inferring schema, computing statistics, running selected analysis tools…",
  diagnosis: "Scoring data quality, detecting duplicates and missing values…",
  intervention: "Generating fix recommendations and simulated impact…",
};

type Props = {
  appState: AppState;
  setAppState: React.Dispatch<React.SetStateAction<AppState>>;
  setActiveStep: (id: Step["id"]) => void;
  markDone: (id: Step["id"]) => void;
};

export function RunningStep({ appState }: Props) {
  const phase = appState.activeStep as "collection" | "diagnosis" | "intervention";

  return (
    <div className="flex flex-col items-center justify-center h-full gap-5 text-center px-8">
      <Loader2 className="w-10 h-10 text-indigo-500 animate-spin" />
      <div>
        <p className="text-base font-semibold text-zinc-800 capitalize">{phase} agent running…</p>
        <p className="text-sm text-zinc-400 mt-1 max-w-sm">{PHASE_DESCRIPTIONS[phase]}</p>
      </div>
    </div>
  );
}
