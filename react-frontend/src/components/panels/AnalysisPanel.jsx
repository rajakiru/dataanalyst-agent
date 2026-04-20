import React, { useState } from 'react'
import { downloadPlotsZip } from '../../api'

export default function AnalysisPanel({ results, sessionId }) {
  const [tab, setTab] = useState('schema')

  const toolResults = results?.tool_results || []
  const byTool = {}
  for (const r of toolResults) {
    if (!byTool[r.tool]) byTool[r.tool] = []
    byTool[r.tool].push(r)
  }

  const plotPaths = results?.plot_paths || []

  const schema = byTool['infer_schema']?.[0]?.result || {}
  const shape = schema.shape || {}
  const roles = schema.roles || {}
  const nullPct = schema.null_pct || {}
  const dtypes = schema.dtypes || {}
  const columns = Object.keys(roles)

  const stats = byTool['summarize_statistics']?.[0]?.result?.statistics || {}
  const statsColumns = Object.keys(stats)

  const correlations = byTool['compute_correlations']?.[0]?.result?.top_correlations || []
  const anomalies = byTool['detect_anomalies'] || []

  const tabs = [
    { id: 'schema', label: 'Schema & Stats' },
    { id: 'viz',    label: 'Visualizations' },
    { id: 'anomaly',label: 'Anomalies' },
  ]

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      {/* Tabs */}
      <div className="flex border-b border-slate-200">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={[
              'px-5 py-3 text-sm font-semibold transition-colors',
              tab === t.id
                ? 'text-indigo-600 border-b-2 border-indigo-600 -mb-px'
                : 'text-slate-500 hover:text-slate-700',
            ].join(' ')}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="p-5">
        {/* SCHEMA & STATS */}
        {tab === 'schema' && (
          <div className="space-y-5">
            <div className="grid grid-cols-2 gap-3">
              <MetricCard label="Rows" value={shape.rows ?? '—'} />
              <MetricCard label="Columns" value={shape.cols ?? '—'} />
            </div>

            {columns.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-2">Schema</h3>
                <div className="overflow-x-auto rounded-lg border border-slate-200">
                  <table className="w-full text-xs">
                    <thead className="bg-slate-50">
                      <tr>
                        {['Column', 'Role', 'Null %', 'Dtype'].map((h) => (
                          <th key={h} className="text-left px-3 py-2 text-slate-500 font-semibold">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {columns.map((col, i) => (
                        <tr key={col} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                          <td className="px-3 py-2 font-medium text-slate-800">{col}</td>
                          <td className="px-3 py-2 text-slate-600">{roles[col]}</td>
                          <td className="px-3 py-2 text-slate-600">{nullPct[col] ?? 0}%</td>
                          <td className="px-3 py-2 text-slate-500">{dtypes[col] || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {statsColumns.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-2">Descriptive Statistics</h3>
                <div className="overflow-x-auto rounded-lg border border-slate-200">
                  <table className="w-full text-xs">
                    <thead className="bg-slate-50">
                      <tr>
                        <th className="text-left px-3 py-2 text-slate-500 font-semibold">Column</th>
                        {Object.keys(stats[statsColumns[0]] || {}).map((k) => (
                          <th key={k} className="text-left px-3 py-2 text-slate-500 font-semibold">{k}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {statsColumns.map((col, i) => (
                        <tr key={col} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                          <td className="px-3 py-2 font-medium text-slate-800">{col}</td>
                          {Object.values(stats[col] || {}).map((v, j) => (
                            <td key={j} className="px-3 py-2 text-slate-600">
                              {typeof v === 'number' ? v.toFixed(3) : String(v ?? '—')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {correlations.length > 0 && (
              <div>
                <h3 className="text-sm font-semibold text-slate-700 mb-2">Top Correlations</h3>
                <div className="overflow-x-auto rounded-lg border border-slate-200">
                  <table className="w-full text-xs">
                    <thead className="bg-slate-50">
                      <tr>
                        {['Column 1', 'Column 2', 'Correlation'].map((h) => (
                          <th key={h} className="text-left px-3 py-2 text-slate-500 font-semibold">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {correlations.map((c, i) => (
                        <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                          <td className="px-3 py-2 text-slate-700">{c.col1}</td>
                          <td className="px-3 py-2 text-slate-700">{c.col2}</td>
                          <td className="px-3 py-2 font-mono text-slate-600">{Number(c.correlation).toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* VISUALIZATIONS */}
        {tab === 'viz' && (
          <div>
            {plotPaths.length === 0 ? (
              <p className="text-sm text-slate-500">No plots were generated.</p>
            ) : (
              <>
                <button
                  onClick={() => downloadPlotsZip(sessionId)}
                  className="mb-4 text-sm bg-indigo-600 text-white px-4 py-2 rounded-lg hover:bg-indigo-700 transition-colors font-semibold"
                >
                  Download all {plotPaths.length} plots (.zip)
                </button>
                <div className="grid grid-cols-2 gap-4">
                  {plotPaths.map((url, i) => (
                    <div key={i} className="rounded-lg overflow-hidden border border-slate-200">
                      <img src={url} alt={url.split('/').pop()} className="w-full object-contain" />
                      <p className="text-xs text-slate-400 px-2 py-1 bg-slate-50 text-center truncate">
                        {url.split('/').pop()}
                      </p>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        )}

        {/* ANOMALIES */}
        {tab === 'anomaly' && (
          <div>
            {anomalies.length === 0 ? (
              <p className="text-sm text-slate-500">No anomaly detection was run.</p>
            ) : (
              <div className="space-y-2">
                {anomalies.map((entry, i) => {
                  const r = entry.result || {}
                  const col = r.column || entry.args?.column || '?'
                  const cnt = r.outlier_count ?? 0
                  const pct = r.outlier_pct ?? 0
                  const bounds = r.iqr_bounds || {}
                  return (
                    <details key={i} className="border border-slate-200 rounded-lg overflow-hidden">
                      <summary className="px-4 py-3 bg-slate-50 text-sm font-semibold text-slate-700 cursor-pointer select-none">
                        {col} — {cnt} outliers ({pct}%)
                      </summary>
                      <div className="px-4 py-3 grid grid-cols-3 gap-3">
                        <MetricCard label="Count" value={cnt} />
                        <MetricCard label="Percent" value={`${pct}%`} />
                        <MetricCard label="IQR Bounds" value={`[${bounds.lower ?? '?'}, ${bounds.upper ?? '?'}]`} />
                        {r.outlier_indices?.length > 0 && (
                          <div className="col-span-3 text-xs text-slate-500">
                            Outlier rows (first 50): {r.outlier_indices.slice(0, 50).join(', ')}
                          </div>
                        )}
                      </div>
                    </details>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function MetricCard({ label, value }) {
  return (
    <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-lg font-bold text-slate-800 mt-0.5">{value}</p>
    </div>
  )
}
