"use client";

import { useRef, useState } from "react";
import { AppState, Step } from "../WorkflowBuilder";
import { UploadCloud } from "lucide-react";

const SAMPLE_DATASETS = [
  { label: "Iris (clean)", value: "iris" },
  { label: "Iris (corrupted)", value: "iris_corrupted" },
  { label: "Titanic", value: "titanic" },
  { label: "Titanic (corrupted)", value: "titanic_corrupted" },
];

type Props = {
  appState: AppState;
  setAppState: React.Dispatch<React.SetStateAction<AppState>>;
  setActiveStep: (id: Step["id"]) => void;
  markDone: (id: Step["id"]) => void;
};

export function UploadStep({ appState, setAppState, setActiveStep, markDone }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);

  function handleFile(file: File) {
    setAppState((p) => ({ ...p, file, sampleDataset: null }));
  }

  function handleSample(value: string) {
    setAppState((p) => ({ ...p, sampleDataset: value, file: null }));
  }

  async function handleAnalyse() {
    const datasetName = appState.file?.name ?? appState.sampleDataset;
    if (!datasetName) return;
    setLoading(true);

    // Call backend plan endpoint
    const formData = new FormData();
    if (appState.file) {
      formData.append("file", appState.file);
    } else if (appState.sampleDataset) {
      formData.append("sample", appState.sampleDataset);
    }

    try {
      const res = await fetch("/api/plan", { method: "POST", body: formData, signal: AbortSignal.timeout(120_000) });
      const planState = await res.json();
      setAppState((p) => ({ ...p, planState }));
      markDone("upload");
      setActiveStep("review");
    } catch {
      alert("Failed to connect to backend. Make sure the Python API is running.");
    } finally {
      setLoading(false);
    }
  }

  const hasSelection = !!(appState.file || appState.sampleDataset);

  return (
    <div className="max-w-2xl mx-auto py-16 px-8">
      <h2 className="text-xl font-semibold text-zinc-900 mb-1">Upload Dataset</h2>
      <p className="text-sm text-zinc-500 mb-8">
        Upload a CSV file or pick a sample dataset to get started.
      </p>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files[0];
          if (f) handleFile(f);
        }}
        className={`flex flex-col items-center justify-center gap-3 border-2 border-dashed rounded-xl py-12 cursor-pointer transition-colors ${
          dragging ? "border-indigo-400 bg-indigo-50" : appState.file ? "border-emerald-400 bg-emerald-50" : "border-zinc-200 hover:border-zinc-300 bg-white"
        }`}
      >
        <UploadCloud className={`w-8 h-8 ${appState.file ? "text-emerald-500" : "text-zinc-400"}`} />
        {appState.file ? (
          <p className="text-sm font-medium text-emerald-700">{appState.file.name}</p>
        ) : (
          <>
            <p className="text-sm text-zinc-600 font-medium">Drop a CSV here or click to browse</p>
            <p className="text-xs text-zinc-400">Any tabular CSV file</p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          className="hidden"
          onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
        />
      </div>

      {/* Sample datasets */}
      <div className="mt-6">
        <p className="text-xs font-medium text-zinc-400 uppercase tracking-widest mb-3">
          Or use a sample dataset
        </p>
        <div className="grid grid-cols-2 gap-2">
          {SAMPLE_DATASETS.map((ds) => (
            <button
              key={ds.value}
              onClick={() => handleSample(ds.value)}
              className={`text-sm px-4 py-2.5 rounded-lg border transition-colors text-left ${
                appState.sampleDataset === ds.value
                  ? "border-indigo-500 bg-indigo-50 text-indigo-700 font-medium"
                  : "border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-50"
              }`}
            >
              {ds.label}
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={handleAnalyse}
        disabled={!hasSelection || loading}
        className="mt-8 w-full py-3 rounded-xl bg-indigo-600 text-white font-medium text-sm hover:bg-indigo-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? "Analysing…" : "Analyse →"}
      </button>
    </div>
  );
}
