import React from 'react'
import { downloadCleanedCsv } from '../../api'

export default function SolutionsPanel({
  results, sessionId,
  appliedSolutions, setAppliedSolutions,
  selectedSolutions, setSelectedSolutions,
}) {

  const solutionsNarrative = results?.solutions_narrative || ''
  const solutionsToolResults = results?.solutions_tool_results || []

  const recommendations = solutionsToolResults
    .filter((r) => r.tool === 'recommend_solutions')
    .flatMap((r) => r.result?.recommendations || [])

  if (!solutionsNarrative && recommendations.length === 0) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <h2 className="text-base font-semibold text-slate-800 mb-2">Suggest Fixes</h2>
        <p className="text-sm text-slate-500">No data quality issues detected — no remediation needed.</p>
      </div>
    )
  }

  function toggleSelect(key) {
    setSelectedSolutions((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  function isApplied(key) {
    return appliedSolutions.some((a) => a.key === key)
  }

  function applyFix(key, label, code) {
    setAppliedSolutions((prev) => [...prev, { key, label, code }])
  }

  function undoFix(key) {
    setAppliedSolutions((prev) => prev.filter((a) => a.key !== key))
  }

  const severityLabel = { high: 'High', medium: 'Medium', low: 'Low' }
  const severityColor = {
    high: 'text-red-600',
    medium: 'text-amber-600',
    low: 'text-emerald-600',
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="p-5 space-y-5">
        <h2 className="text-base font-semibold text-slate-800">Suggest Fixes</h2>

        {/* Applied banner */}
        {appliedSolutions.length > 0 && (
          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 flex items-center justify-between">
            <span className="text-sm text-indigo-700 font-semibold">
              {appliedSolutions.length} fix(es) applied
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => downloadCleanedCsv(sessionId, appliedSolutions)}
                className="text-xs bg-indigo-600 text-white px-3 py-1.5 rounded-lg font-semibold hover:bg-indigo-700 transition-colors"
              >
                Download cleaned CSV
              </button>
              <button
                onClick={() => setAppliedSolutions([])}
                className="text-xs border border-indigo-200 text-indigo-600 px-3 py-1.5 rounded-lg hover:bg-indigo-100 transition-colors"
              >
                Clear all
              </button>
            </div>
          </div>
        )}

        {/* Remediation strategy */}
        {solutionsNarrative && (
          <details open className="border border-slate-200 rounded-xl overflow-hidden">
            <summary className="px-4 py-3 bg-slate-50 text-sm font-semibold text-slate-700 cursor-pointer select-none">
              Remediation Strategy
            </summary>
            <div className="px-4 py-3 text-sm text-slate-600 whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
              {solutionsNarrative}
            </div>
          </details>
        )}

        {/* Per-issue recommendations */}
        {recommendations.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-1">Actionable Solutions by Issue</h3>
            <p className="text-xs text-slate-400 mb-3">
              Check actions to include in report. Click Apply Fix to run the transformation.
            </p>

            <div className="space-y-3">
              {recommendations.map((rec, ri) => {
                const colName = rec.column || '—'
                const issueType = rec.issue_type || 'unknown'
                const severity = rec.severity || 'medium'
                const actions = rec.actions || []

                return (
                  <details key={ri} className="border border-slate-200 rounded-xl overflow-hidden">
                    <summary className="px-4 py-3 bg-slate-50 text-sm font-semibold text-slate-700 cursor-pointer select-none flex items-center gap-2">
                      <span className="flex-1">{colName} — {issueType}</span>
                      <span className={`text-xs font-normal ${severityColor[severity] || 'text-slate-500'}`}>
                        {severityLabel[severity] || severity}
                      </span>
                    </summary>

                    <div className="px-4 py-3 space-y-4">
                      {actions.map((action, ai) => {
                        const key = `${colName}|${issueType}|${action.action || ''}`
                        const applied = isApplied(key)
                        const selected = selectedSolutions.has(key)
                        const priority = (action.priority || 'medium').toUpperCase()

                        return (
                          <div key={ai} className="space-y-1">
                            <div className="flex items-center gap-3">
                              <input
                                type="checkbox"
                                checked={selected}
                                onChange={() => toggleSelect(key)}
                                className="w-4 h-4 accent-indigo-600 flex-shrink-0"
                              />
                              <span className="text-sm font-semibold text-slate-700 flex-1">
                                [{priority}] {action.action}
                                {applied && <span className="ml-2 text-emerald-600 font-normal text-xs">— applied</span>}
                              </span>
                              {action.implementation && (
                                <button
                                  onClick={() => applied ? undoFix(key) : applyFix(key, action.action, action.implementation)}
                                  className={[
                                    'text-xs px-3 py-1.5 rounded-lg font-semibold transition-colors flex-shrink-0',
                                    applied
                                      ? 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                                      : 'bg-indigo-600 text-white hover:bg-indigo-700',
                                  ].join(' ')}
                                >
                                  {applied ? 'Undo' : 'Apply Fix'}
                                </button>
                              )}
                            </div>

                            {action.rationale && (
                              <p className="text-xs text-slate-500 ml-7">
                                <strong>Rationale:</strong> {action.rationale}
                              </p>
                            )}

                            {action.implementation && (
                              <pre className="ml-7 bg-slate-900 text-slate-100 text-xs p-3 rounded-lg overflow-x-auto">
                                <code>{action.implementation}</code>
                              </pre>
                            )}

                            {ai < actions.length - 1 && <hr className="border-slate-100 ml-7 mt-2" />}
                          </div>
                        )
                      })}
                    </div>
                  </details>
                )
              })}
            </div>
          </div>
        )}

        {selectedSolutions.size > 0 && (
          <div className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-xs text-slate-500">
            {selectedSolutions.size} action(s) selected — the exported report will include only these.
          </div>
        )}
      </div>
    </div>
  )
}
