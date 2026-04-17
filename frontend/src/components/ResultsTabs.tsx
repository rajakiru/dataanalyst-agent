"use client";

import { useState } from "react";

type Tab = "schema" | "quality" | "visualizations" | "solutions" | "report";

const TABS: { id: Tab; label: string }[] = [
  { id: "schema", label: "Schema & Stats" },
  { id: "quality", label: "Quality" },
  { id: "visualizations", label: "Visualizations" },
  { id: "solutions", label: "Solutions" },
  { id: "report", label: "Report" },
];

type Props = {
  results: Record<string, unknown>;
  onReset: () => void;
};

export function ResultsTabs({ results, onReset }: Props) {
  const [active, setActive] = useState<Tab>("schema");

  return (
    <div className="flex flex-col h-full">
      {/* Tab bar */}
      <div className="flex border-b border-zinc-200 bg-white px-6 gap-1 pt-4">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setActive(t.id)}
            className={`px-4 py-2 text-sm rounded-t-lg transition-colors ${
              active === t.id
                ? "bg-indigo-50 text-indigo-700 font-medium border-b-2 border-indigo-600"
                : "text-zinc-500 hover:text-zinc-700"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-y-auto px-8 py-6">
        {active === "schema" && <SchemaTab results={results} />}
        {active === "quality" && <QualityTab results={results} />}
        {active === "visualizations" && <VisualizationsTab results={results} />}
        {active === "solutions" && <SolutionsTab results={results} />}
        {active === "report" && <ReportTab results={results} onReset={onReset} />}
      </div>
    </div>
  );
}

function SchemaTab({ results }: { results: Record<string, unknown> }) {
  const schema = results.schema_result as Record<string, unknown> | undefined;
  if (!schema) return <Empty message="No schema data available." />;

  return (
    <div>
      <h3 className="text-base font-semibold text-zinc-900 mb-4">Schema & Statistics</h3>
      <pre className="text-xs bg-zinc-50 border border-zinc-200 rounded-xl p-5 overflow-auto text-zinc-700 whitespace-pre-wrap">
        {JSON.stringify(schema, null, 2)}
      </pre>
    </div>
  );
}

function QualityTab({ results }: { results: Record<string, unknown> }) {
  const quality = results.quality_score as Record<string, unknown> | undefined;
  const issues = (results.quality_tool_results as unknown[]) ?? [];

  return (
    <div className="flex flex-col gap-6">
      {quality && (
        <div>
          <h3 className="text-base font-semibold text-zinc-900 mb-3">Quality Score</h3>
          <pre className="text-xs bg-zinc-50 border border-zinc-200 rounded-xl p-5 overflow-auto text-zinc-700 whitespace-pre-wrap">
            {JSON.stringify(quality, null, 2)}
          </pre>
        </div>
      )}
      {issues.length > 0 && (
        <div>
          <h3 className="text-base font-semibold text-zinc-900 mb-3">Detected Issues</h3>
          {issues.map((r: unknown, i: number) => (
            <pre key={i} className="text-xs bg-zinc-50 border border-zinc-200 rounded-xl p-5 mb-3 overflow-auto text-zinc-700 whitespace-pre-wrap">
              {JSON.stringify(r, null, 2)}
            </pre>
          ))}
        </div>
      )}
      {!quality && issues.length === 0 && <Empty message="No quality data available." />}
    </div>
  );
}

function VisualizationsTab({ results }: { results: Record<string, unknown> }) {
  const plots = (results.plots as string[]) ?? [];
  if (plots.length === 0) return <Empty message="No visualizations generated." />;

  return (
    <div>
      <h3 className="text-base font-semibold text-zinc-900 mb-4">Visualizations</h3>
      <div className="grid grid-cols-2 gap-4">
        {plots.map((src, i) => (
          <img
            key={i}
            src={`data:image/png;base64,${src}`}
            alt={`Plot ${i + 1}`}
            className="rounded-xl border border-zinc-200 w-full"
          />
        ))}
      </div>
    </div>
  );
}

function SolutionsTab({ results }: { results: Record<string, unknown> }) {
  const solutions = (results.solutions_tool_results as unknown[]) ?? [];
  if (solutions.length === 0) return <Empty message="No solutions generated." />;

  return (
    <div>
      <h3 className="text-base font-semibold text-zinc-900 mb-4">Recommended Solutions</h3>
      {solutions.map((s: unknown, i: number) => (
        <pre key={i} className="text-xs bg-zinc-50 border border-zinc-200 rounded-xl p-5 mb-3 overflow-auto text-zinc-700 whitespace-pre-wrap">
          {JSON.stringify(s, null, 2)}
        </pre>
      ))}
    </div>
  );
}

function ReportTab({ results, onReset }: { results: Record<string, unknown>; onReset: () => void }) {
  const narrative = (results.narrative as string) ?? "";
  const report = (results.markdown_report as string) ?? narrative;

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-base font-semibold text-zinc-900">Report</h3>
        <div className="flex gap-2">
          {report && (
            <a
              href={`data:text/markdown;charset=utf-8,${encodeURIComponent(report)}`}
              download="autoanalyst_report.md"
              className="text-xs px-3 py-1.5 rounded-lg border border-zinc-200 text-zinc-600 hover:bg-zinc-50 transition-colors"
            >
              Download .md
            </a>
          )}
          <button
            onClick={onReset}
            className="text-xs px-3 py-1.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors"
          >
            ← New analysis
          </button>
        </div>
      </div>
      {report ? (
        <div className="prose prose-sm max-w-none">
          <pre className="text-sm text-zinc-700 whitespace-pre-wrap leading-relaxed bg-white border border-zinc-200 rounded-xl p-6">
            {report}
          </pre>
        </div>
      ) : (
        <Empty message="No report generated." />
      )}
    </div>
  );
}

function Empty({ message }: { message: string }) {
  return (
    <div className="flex items-center justify-center h-40 text-sm text-zinc-400">
      {message}
    </div>
  );
}
