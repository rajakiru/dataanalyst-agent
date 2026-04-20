import React, { useState, useEffect } from 'react'
import { X, ArrowRight, Loader2 } from 'lucide-react'
import { previewFix } from '../api'

export default function FixPreviewModal({ sessionId, fix, onClose }) {
  const [loading, setLoading] = useState(true)
  const [data, setData]       = useState(null)
  const [error, setError]     = useState(null)
  const [view, setView]       = useState('diff') // 'diff' | 'before' | 'after'

  useEffect(() => {
    setLoading(true)
    setError(null)
    previewFix(sessionId, fix.code, fix.column || '')
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [sessionId, fix.code, fix.column])

  const changedSet = new Set(
    (data?.changed_cells || []).map(c => `${c.row}|${c.col}`)
  )

  function cellClass(rowIdx, col, isBefore) {
    const changed = changedSet.has(`${rowIdx}|${col}`)
    if (!changed) return 'px-3 py-1.5 text-xs text-slate-600'
    return isBefore
      ? 'px-3 py-1.5 text-xs font-semibold bg-red-50 text-red-700'
      : 'px-3 py-1.5 text-xs font-semibold bg-emerald-50 text-emerald-700'
  }

  function fmtVal(v) {
    if (v === null || v === undefined) return <span className="italic text-slate-300">null</span>
    if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(4)
    return String(v)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-4xl max-h-[85vh] flex flex-col overflow-hidden">

        {/* Header */}
        <div className="flex items-start justify-between px-6 py-4 border-b border-slate-200 flex-shrink-0">
          <div>
            <h2 className="text-base font-semibold text-slate-800">Fix Preview</h2>
            <p className="text-sm text-slate-500 mt-0.5">{fix.label}</p>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-slate-100 text-slate-400">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Code */}
        <div className="px-6 py-3 border-b border-slate-100 bg-slate-900 flex-shrink-0">
          <pre className="text-xs text-slate-200 overflow-x-auto leading-relaxed">
            <code>{fix.code}</code>
          </pre>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4">
          {loading && (
            <div className="flex flex-col items-center justify-center gap-3 py-16 text-slate-400">
              <Loader2 className="w-8 h-8 animate-spin text-indigo-500" />
              <p className="text-sm">Applying fix to sample rows...</p>
            </div>
          )}

          {error && (
            <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {data && !loading && (
            <>
              {data.exec_error && (
                <div className="mb-4 rounded-lg bg-amber-50 border border-amber-200 px-4 py-3 text-sm text-amber-700">
                  <strong>Warning:</strong> {data.exec_error}
                </div>
              )}

              {/* Stats row */}
              <div className="flex items-center gap-4 mb-4">
                <span className="text-sm text-slate-500">
                  Showing <strong>{data.before.length}</strong> of{' '}
                  <strong>{data.total_affected}</strong> affected rows
                </span>
                <span className="text-sm text-slate-500">
                  <strong>{data.changed_cells.length}</strong> cells changed
                </span>
                {/* View toggle */}
                <div className="ml-auto flex rounded-lg border border-slate-200 overflow-hidden text-xs">
                  {['diff', 'before', 'after'].map(v => (
                    <button
                      key={v}
                      onClick={() => setView(v)}
                      className={[
                        'px-3 py-1.5 font-medium transition-colors capitalize',
                        view === v
                          ? 'bg-indigo-600 text-white'
                          : 'bg-white text-slate-600 hover:bg-slate-50',
                      ].join(' ')}
                    >
                      {v}
                    </button>
                  ))}
                </div>
              </div>

              {/* Diff view: side-by-side */}
              {view === 'diff' && (
                <div className="grid grid-cols-[1fr_auto_1fr] gap-2">
                  {/* Before table */}
                  <div>
                    <div className="text-xs font-semibold text-red-600 mb-1.5 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-red-400 inline-block" />
                      Before
                    </div>
                    <TableView
                      columns={data.columns}
                      rows={data.before}
                      cellClass={(ri, col) => cellClass(ri, col, true)}
                      fmtVal={fmtVal}
                    />
                  </div>

                  {/* Arrow */}
                  <div className="flex items-center pt-8">
                    <ArrowRight className="w-5 h-5 text-slate-300 flex-shrink-0" />
                  </div>

                  {/* After table */}
                  <div>
                    <div className="text-xs font-semibold text-emerald-600 mb-1.5 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
                      After
                    </div>
                    <TableView
                      columns={data.columns}
                      rows={data.after}
                      cellClass={(ri, col) => cellClass(ri, col, false)}
                      fmtVal={fmtVal}
                    />
                  </div>
                </div>
              )}

              {/* Single-table views */}
              {view === 'before' && (
                <TableView
                  columns={data.columns}
                  rows={data.before}
                  cellClass={(ri, col) => cellClass(ri, col, true)}
                  fmtVal={fmtVal}
                />
              )}
              {view === 'after' && (
                <TableView
                  columns={data.columns}
                  rows={data.after}
                  cellClass={(ri, col) => cellClass(ri, col, false)}
                  fmtVal={fmtVal}
                />
              )}

              {/* Legend */}
              <div className="mt-4 flex items-center gap-4 text-xs text-slate-400">
                <span className="flex items-center gap-1">
                  <span className="inline-block w-3 h-3 rounded bg-red-100 border border-red-200" />
                  Changed (before)
                </span>
                <span className="flex items-center gap-1">
                  <span className="inline-block w-3 h-3 rounded bg-emerald-100 border border-emerald-200" />
                  Changed (after)
                </span>
                <span className="italic text-slate-300">null = missing value</span>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-100 flex-shrink-0 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-slate-600 border border-slate-200 rounded-lg hover:bg-slate-50"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

function TableView({ columns, rows, cellClass, fmtVal }) {
  return (
    <div className="overflow-auto rounded-lg border border-slate-200 text-xs">
      <table className="w-full border-collapse">
        <thead className="bg-slate-50 sticky top-0">
          <tr>
            <th className="px-3 py-2 text-left text-slate-400 font-semibold w-8">#</th>
            {columns.map(c => (
              <th key={c} className="px-3 py-2 text-left text-slate-600 font-semibold whitespace-nowrap">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri} className={ri % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
              <td className="px-3 py-1.5 text-slate-300">{ri + 1}</td>
              {columns.map(col => (
                <td key={col} className={cellClass(ri, col)}>
                  {fmtVal(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
