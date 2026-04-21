import { useState, useEffect } from 'react'
import { X, ArrowRight, Loader2, Sparkles, RefreshCw } from 'lucide-react'
import { previewFix } from '../api'

// ---------------------------------------------------------------------------
// Image generation preview (DALL-E 2 fix)
// Shows: cascade banner + before/after NaN rows + pre-generated synthetic image
// ---------------------------------------------------------------------------
function ImageGenerationPreview({ sessionId, fix, onClose }) {
  const [loading, setLoading]     = useState(true)
  const [data, setData]           = useState(null)
  const [error, setError]         = useState(null)
  const [flowerType, setFlowerType] = useState('daisy')

  const flowerTypes = ['daisy', 'lavender', 'rose', 'sunflower', 'tulip']

  function fetchPreview(flower) {
    setFlowerType(flower)
    setLoading(true)
    setError(null)
    previewFix(sessionId, fix.code || '', fix.column || '', 15, 'image_generation', flower)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchPreview('daisy') }, [])

  const changedSet = new Set((data?.changed_cells || []).map(c => `${c.row}|${c.col}`))

  function cellClass(rowIdx, col, isBefore) {
    if (!changedSet.has(`${rowIdx}|${col}`)) return 'px-2 py-1.5 text-xs text-slate-600'
    return isBefore
      ? 'px-2 py-1.5 text-xs font-semibold bg-red-50 text-red-700'
      : 'px-2 py-1.5 text-xs font-semibold bg-emerald-50 text-emerald-700'
  }

  function fmtVal(v) {
    if (v === null || v === undefined) return <span className="italic text-slate-300">null</span>
    if (typeof v === 'number') return Number.isInteger(v) ? v : v.toFixed(1)
    return String(v)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[90vh] flex flex-col overflow-hidden">

        {/* Header */}
        <div className="flex items-start justify-between px-6 py-4 border-b border-slate-200 flex-shrink-0">
          <div>
            <div className="flex items-center gap-2 mb-0.5">
              <Sparkles className="w-4 h-4 text-indigo-500" />
              <h2 className="text-base font-semibold text-slate-800">DALL-E 2 Fix Preview</h2>
            </div>
            <p className="text-sm text-slate-500">{fix.label}</p>
          </div>
          <button onClick={onClose} className="p-1 rounded-lg hover:bg-slate-100 text-slate-400">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Cascade banner */}
        <div className="px-6 py-3 bg-amber-50 border-b border-amber-100 flex-shrink-0">
          <p className="text-xs text-amber-800 leading-relaxed">
            <strong>Data Cascade:</strong> 4 missing images → feature extraction fails → 4 NaN rows
            in feature table → silent class bias in downstream model.
            DALL-E 2 generates synthetic replacements to restore 80% → 100% coverage.
          </p>
        </div>

        {/* Flower selector */}
        <div className="px-6 py-2.5 border-b border-slate-100 flex items-center gap-3 flex-shrink-0">
          <span className="text-xs text-slate-500 font-medium flex-shrink-0">Preview missing:</span>
          <div className="flex gap-1.5 flex-wrap">
            {flowerTypes.map(f => (
              <button
                key={f}
                onClick={() => fetchPreview(f)}
                disabled={loading}
                className={[
                  'text-xs px-2.5 py-1 rounded-full border capitalize transition-colors disabled:opacity-40',
                  f === flowerType
                    ? 'bg-indigo-600 text-white border-indigo-600'
                    : 'border-indigo-200 text-indigo-700 bg-indigo-50 hover:bg-indigo-100',
                ].join(' ')}
              >
                {f}
              </button>
            ))}
          </div>
          <button
            onClick={() => fetchPreview(flowerTypes[Math.floor(Math.random() * flowerTypes.length)])}
            disabled={loading}
            className="ml-auto p-1.5 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50 disabled:opacity-40"
            title="Random flower"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
          {loading && (
            <div className="flex flex-col items-center justify-center gap-3 py-12 text-slate-400">
              <Loader2 className="w-7 h-7 animate-spin text-indigo-500" />
              <p className="text-sm">Loading fix preview…</p>
            </div>
          )}

          {error && (
            <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-700">
              {error}
            </div>
          )}

          {data && !loading && (
            <>
              {/* Before / After diff table */}
              <div>
                <p className="text-xs text-slate-500 mb-2">
                  <strong>{data.total_affected}</strong> NaN rows detected — showing feature columns before and after fix
                </p>
                <div className="grid grid-cols-[1fr_auto_1fr] gap-2">
                  {/* Before */}
                  <div>
                    <div className="text-xs font-semibold text-red-600 mb-1.5 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-red-400 inline-block" />
                      Before (NaN — extraction failed)
                    </div>
                    <div className="overflow-auto rounded-lg border border-slate-200">
                      <table className="w-full border-collapse text-xs">
                        <thead className="bg-slate-50">
                          <tr>
                            {data.columns.map(c => (
                              <th key={c} className="px-2 py-1.5 text-left text-slate-500 font-semibold whitespace-nowrap">{c}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {data.before.map((row, ri) => (
                            <tr key={ri} className={ri % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                              {data.columns.map(col => (
                                <td key={col} className={cellClass(ri, col, true)}>{fmtVal(row[col])}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Arrow */}
                  <div className="flex items-center pt-8">
                    <ArrowRight className="w-5 h-5 text-slate-300 flex-shrink-0" />
                  </div>

                  {/* After */}
                  <div>
                    <div className="text-xs font-semibold text-emerald-600 mb-1.5 flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
                      After (DALL-E features extracted)
                    </div>
                    <div className="overflow-auto rounded-lg border border-slate-200">
                      <table className="w-full border-collapse text-xs">
                        <thead className="bg-slate-50">
                          <tr>
                            {data.columns.map(c => (
                              <th key={c} className="px-2 py-1.5 text-left text-slate-500 font-semibold whitespace-nowrap">{c}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {data.after.map((row, ri) => (
                            <tr key={ri} className={ri % 2 === 0 ? 'bg-white' : 'bg-slate-50'}>
                              {data.columns.map(col => (
                                <td key={col} className={cellClass(ri, col, false)}>{fmtVal(row[col])}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>

                {/* Legend */}
                <div className="mt-2 flex items-center gap-4 text-xs text-slate-400">
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-3 h-3 rounded bg-red-100 border border-red-200" />
                    NaN (before)
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="inline-block w-3 h-3 rounded bg-emerald-100 border border-emerald-200" />
                    Filled (after)
                  </span>
                </div>
              </div>

              {/* Generated image */}
              {data.image_b64 && (
                <div className="flex flex-col items-center gap-3 pt-2 border-t border-slate-100">
                  <p className="text-xs text-slate-500 self-start">
                    Synthetic replacement image for <strong>{flowerType}</strong>:
                  </p>
                  <div className="relative">
                    <img
                      src={`data:image/png;base64,${data.image_b64}`}
                      alt={`DALL-E generated ${flowerType}`}
                      className="w-48 h-48 rounded-xl border-2 border-indigo-200 shadow-md object-cover"
                    />
                    <div className="absolute -top-2 -right-2 bg-indigo-600 text-white text-xs px-2 py-0.5 rounded-full font-semibold">
                      DALL-E 2
                    </div>
                  </div>
                  <p className="text-xs text-slate-400 italic text-center max-w-xs">
                    "{data.image_prompt}"
                  </p>
                  <div className="flex items-center gap-2 text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 rounded-lg px-3 py-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
                    Image saved → features extracted → NaN row resolved → cascade fixed
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-100 flex-shrink-0 flex justify-between items-center">
          <p className="text-xs text-slate-400">~$0.02/image · 256×256 · DALL-E 2</p>
          <button onClick={onClose} className="px-4 py-2 text-sm font-medium text-slate-600 border border-slate-200 rounded-lg hover:bg-slate-50">
            Close
          </button>
        </div>
      </div>
    </div>
  )
}

export default function FixPreviewModal({ sessionId, fix, onClose }) {
  // Route to the appropriate modal based on fix type
  if (fix.fix_type === 'image_generation') {
    return <ImageGenerationPreview sessionId={sessionId} fix={fix} onClose={onClose} />
  }
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
