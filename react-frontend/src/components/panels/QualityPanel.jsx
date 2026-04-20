import React, { useState } from 'react'

export default function QualityPanel({ results }) {
  const [dismissed, setDismissed] = useState(new Set())

  const toolResults = results?.tool_results || []
  const qToolResults = results?.quality_tool_results || []
  const qPlotPaths = results?.quality_plot_paths || []
  const qNarrative = results?.quality_narrative || ''

  const qByTool = {}
  for (const r of qToolResults) {
    if (!qByTool[r.tool]) qByTool[r.tool] = []
    qByTool[r.tool].push(r)
  }

  const byTool = {}
  for (const r of toolResults) {
    if (!byTool[r.tool]) byTool[r.tool] = []
    byTool[r.tool].push(r)
  }

  const dq = qByTool['compute_data_quality_score']?.[0]?.result || null

  if (!dq) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <h2 className="text-base font-semibold text-slate-800 mb-2">Detect Issues</h2>
        <p className="text-sm text-slate-500">Data quality audit did not run.</p>
      </div>
    )
  }

  const overall = dq.overall_score ?? 0
  const breakdown = dq.breakdown || {}
  const issues = dq.issues || []
  const grade = overall >= 90 ? 'Excellent' : overall >= 75 ? 'Good' : overall >= 55 ? 'Fair' : 'Poor'

  const dupResult = qByTool['detect_duplicates']?.[0]?.result || {}
  const dupCount = dupResult.duplicate_rows ?? 0

  const schema = byTool['infer_schema']?.[0]?.result || {}
  const roles = schema.roles || {}
  const nullPcts = schema.null_pct || {}
  const issueColSet = new Set(issues.map((i) => i.column))

  const columnRows = Object.keys(roles).map((col) => {
    const nullP = nullPcts[col] ?? 0
    let status = 'OK'
    if (nullP > 50 || issues.some((i) => i.column === col && i.severity === 'high')) {
      status = 'High'
    } else if (nullP > 0 || issueColSet.has(col)) {
      status = 'Medium'
    }
    return { col, role: roles[col], complete: (100 - nullP).toFixed(1), missing: nullP.toFixed(1), status }
  })

  function toggleDismiss(key) {
    setDismissed((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const heatmapPaths = qPlotPaths.filter((p) => p.includes('missing'))
  const qqPaths = qPlotPaths.filter((p) => p.includes('qq_'))

  const statusColor = {
    OK: 'text-emerald-600 bg-emerald-50',
    Medium: 'text-amber-600 bg-amber-50',
    High: 'text-red-600 bg-red-50',
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="p-5 space-y-5">
        <h2 className="text-base font-semibold text-slate-800">Detect Issues</h2>

        {/* Score */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex items-baseline gap-2 mb-1">
              <span className="text-3xl font-extrabold text-slate-800">{overall}</span>
              <span className="text-sm text-slate-500">/ 100</span>
              <span className="text-sm font-semibold text-slate-500">{grade}</span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full bg-indigo-500 transition-all"
                style={{ width: `${Math.max(0, Math.min(100, overall))}%` }}
              />
            </div>
          </div>

          <div className="space-y-2">
            {[
              ['Completeness', breakdown.completeness ?? 0, 'Non-null values'],
              ['Uniqueness',   breakdown.uniqueness ?? 0,   'Non-duplicate rows'],
              ['Consistency',  breakdown.consistency ?? 0,  'No mixed types'],
            ].map(([lbl, val, tip]) => (
              <div key={lbl}>
                <div className="flex justify-between text-xs mb-0.5">
                  <span className="font-semibold text-slate-600">{lbl}</span>
                  <span className="text-slate-400">{Number(val).toFixed(1)}%</span>
                </div>
                <div className="w-full h-1.5 bg-slate-200 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full bg-indigo-400"
                    style={{ width: `${Math.max(0, Math.min(100, val))}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <hr className="border-slate-200" />

        {/* Column health */}
        {columnRows.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-2">Column Health</h3>
            <div className="overflow-x-auto rounded-lg border border-slate-200">
              <table className="w-full text-xs">
                <thead className="bg-slate-50">
                  <tr>
                    {['Column', 'Type', 'Complete%', 'Missing%', 'Status'].map((h) => (
                      <th key={h} className="text-left px-3 py-2 text-slate-500 font-semibold">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {columnRows.map((row, i) => (
                    <tr key={row.col} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                      <td className="px-3 py-2 font-medium text-slate-800">{row.col}</td>
                      <td className="px-3 py-2 text-slate-600">{row.role}</td>
                      <td className="px-3 py-2 text-slate-600">{row.complete}%</td>
                      <td className="px-3 py-2 text-slate-600">{row.missing}%</td>
                      <td className="px-3 py-2">
                        <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${statusColor[row.status] || ''}`}>
                          {row.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        <hr className="border-slate-200" />

        {/* Issues list */}
        <div>
          <h3 className="text-sm font-semibold text-slate-700 mb-2">
            Issues ({issues.length + (dupCount > 0 ? 1 : 0)} total)
          </h3>
          <p className="text-xs text-slate-400 mb-3">Dismiss issues already handled — the Solutions node will skip them.</p>

          {issues.length === 0 && dupCount === 0 ? (
            <div className="bg-emerald-50 border border-emerald-200 rounded-lg px-4 py-3 text-sm text-emerald-700">
              No issues detected — this dataset looks clean.
            </div>
          ) : (
            <div className="space-y-2">
              {issues.map((issue, i) => {
                const key = `${issue.column}|${issue.issue}`
                const isDismissed = dismissed.has(key)
                return (
                  <div key={i} className="flex items-start gap-3">
                    <div className="flex-1">
                      {isDismissed ? (
                        <p className="text-sm text-slate-400 line-through">
                          {issue.column} — {issue.issue}
                        </p>
                      ) : issue.severity === 'high' ? (
                        <div className="bg-red-50 border border-red-200 rounded-lg px-3 py-2 text-sm text-red-700">
                          <strong>{issue.column}</strong> — {issue.issue}
                        </div>
                      ) : (
                        <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-sm text-amber-700">
                          <strong>{issue.column}</strong> — {issue.issue}
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => toggleDismiss(key)}
                      className="text-xs px-2 py-1 rounded border border-slate-200 text-slate-500 hover:bg-slate-50 flex-shrink-0 mt-0.5"
                    >
                      {isDismissed ? 'Restore' : 'Dismiss'}
                    </button>
                  </div>
                )
              })}

              {dupCount > 0 && (() => {
                const key = 'duplicates|duplicate_rows'
                const isDismissed = dismissed.has(key)
                return (
                  <div className="flex items-start gap-3">
                    <div className="flex-1">
                      {isDismissed ? (
                        <p className="text-sm text-slate-400 line-through">
                          Duplicate rows — {dupCount} rows
                        </p>
                      ) : (
                        <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-sm text-amber-700">
                          <strong>Duplicate rows</strong> — {dupCount} rows ({dupResult.duplicate_pct ?? 0}%) are exact duplicates.
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => toggleDismiss(key)}
                      className="text-xs px-2 py-1 rounded border border-slate-200 text-slate-500 hover:bg-slate-50 flex-shrink-0 mt-0.5"
                    >
                      {isDismissed ? 'Restore' : 'Dismiss'}
                    </button>
                  </div>
                )
              })()}
            </div>
          )}
        </div>

        {/* Plots */}
        {heatmapPaths.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-2">Missing Value Map</h3>
            {heatmapPaths.map((url, i) => (
              <img key={i} src={url} alt="Missing heatmap" className="w-full rounded-lg border border-slate-200" />
            ))}
          </div>
        )}

        {qqPaths.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-2">Q-Q Plots</h3>
            <p className="text-xs text-slate-400 mb-2">Points close to the line indicate a normal distribution.</p>
            <div className="grid grid-cols-3 gap-3">
              {qqPaths.map((url, i) => {
                const fname = url.split('/').pop().replace('qq_', '').replace('.png', '')
                return (
                  <div key={i} className="rounded-lg overflow-hidden border border-slate-200">
                    <img src={url} alt={fname} className="w-full object-contain" />
                    <p className="text-xs text-center text-slate-400 py-1 bg-slate-50">{fname}</p>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Quality narrative */}
        {qNarrative && (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-2">Quality Assessment</h3>
            <p className="text-sm text-slate-600 whitespace-pre-wrap leading-relaxed">{qNarrative}</p>
          </div>
        )}
      </div>
    </div>
  )
}
