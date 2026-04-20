import React, { useState } from 'react'
import { runExecute, getResults } from '../../api'

const ALWAYS_ON_TOOLS = new Set([
  'compute_data_quality_score', 'detect_duplicates',
  'plot_missing_heatmap', 'plot_qq', 'recommend_solutions', 'infer_schema',
])

export default function ReviewPanel({
  sessionId,
  planData,
  model,
  setStep,
  setSelectedNode,
  setApprovedTools,
  setEvents,
  setResults,
  setError,
  reset,
}) {
  const availableTools = (planData?.available_tool_names || []).filter(
    (t) => !ALWAYS_ON_TOOLS.has(t)
  )
  const plannedTools = (planData?.planned_tool_names || []).filter(
    (t) => !ALWAYS_ON_TOOLS.has(t)
  )
  const toolDescriptions = planData?.tool_descriptions || {}
  const analysisPlan = planData?.analysis_plan || ''

  const [checked, setChecked] = useState(() => {
    const init = {}
    for (const t of availableTools) {
      init[t] = plannedTools.includes(t)
    }
    return init
  })

  const approved = availableTools.filter((t) => checked[t])

  function toggle(tool) {
    setChecked((prev) => ({ ...prev, [tool]: !prev[tool] }))
  }

  async function handleRun() {
    setApprovedTools(approved)
    setEvents([])
    setStep('running')
    setSelectedNode('analysis')

    try {
      const collected = []
      await runExecute(sessionId, approved, model, (event) => {
        collected.push(event)
        setEvents((prev) => [...prev, event])
      })
      const results = await getResults(sessionId)
      setResults(results)
      setStep('complete')
      setSelectedNode('analysis')
    } catch (e) {
      setError(e.message || 'Execution failed')
      setStep('review')
    }
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
      <h2 className="text-base font-semibold text-slate-800 mb-4">Analysis Node — Review Plan</h2>

      {analysisPlan && (
        <details open className="mb-4 border border-slate-200 rounded-xl overflow-hidden">
          <summary className="px-4 py-3 text-sm font-semibold text-slate-700 bg-slate-50 cursor-pointer select-none">
            Agent's Analysis Plan
          </summary>
          <div className="px-4 py-3 text-sm text-slate-600 whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
            {analysisPlan}
          </div>
        </details>
      )}

      <hr className="border-slate-200 mb-4" />

      <p className="text-xs text-slate-500 mb-3">
        Pre-checked based on the agent's plan. Uncheck tools to skip them.
      </p>

      <div className="grid grid-cols-2 gap-2 mb-6">
        {availableTools.map((tool) => {
          const desc = toolDescriptions[tool] || ''
          const shortDesc = desc.split('.')[0] || ''
          return (
            <label key={tool} className="flex flex-col gap-0.5 cursor-pointer">
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={!!checked[tool]}
                  onChange={() => toggle(tool)}
                  className="w-4 h-4 accent-indigo-600"
                />
                <span className="text-sm font-semibold text-slate-700">{tool}</span>
              </div>
              {shortDesc && (
                <p className="text-xs text-slate-400 pl-6">{shortDesc}</p>
              )}
            </label>
          )
        })}
      </div>

      <div className="flex gap-3">
        <button
          onClick={handleRun}
          disabled={approved.length === 0}
          className="flex-1 bg-indigo-600 text-white text-sm font-semibold py-2.5 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Run Analysis
        </button>
        <button
          onClick={reset}
          className="px-4 py-2.5 text-sm font-semibold text-slate-600 border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors"
        >
          Start Over
        </button>
      </div>
    </div>
  )
}
