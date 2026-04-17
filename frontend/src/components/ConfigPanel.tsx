"use client";

import { useRef, useState } from "react";
import { AppState, Step } from "./WorkflowBuilder";
import { UploadStep } from "./steps/UploadStep";
import { ReviewStep } from "./steps/ReviewStep";
import { RunningStep } from "./steps/RunningStep";

type Props = {
  appState: AppState;
  setAppState: React.Dispatch<React.SetStateAction<AppState>>;
  setActiveStep: (id: Step["id"]) => void;
  markDone: (id: Step["id"]) => void;
};

export function ConfigPanel({ appState, setAppState, setActiveStep, markDone }: Props) {
  if (appState.activeStep === "upload") {
    return (
      <UploadStep
        appState={appState}
        setAppState={setAppState}
        setActiveStep={setActiveStep}
        markDone={markDone}
      />
    );
  }

  if (appState.activeStep === "review") {
    return (
      <ReviewStep
        appState={appState}
        setAppState={setAppState}
        setActiveStep={setActiveStep}
        markDone={markDone}
      />
    );
  }

  if (
    appState.activeStep === "collection" ||
    appState.activeStep === "diagnosis" ||
    appState.activeStep === "intervention"
  ) {
    return (
      <RunningStep
        appState={appState}
        setAppState={setAppState}
        setActiveStep={setActiveStep}
        markDone={markDone}
      />
    );
  }

  return (
    <div className="flex items-center justify-center h-full text-zinc-400 text-sm">
      Select a step to configure
    </div>
  );
}
