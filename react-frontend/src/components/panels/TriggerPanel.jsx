import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Upload } from 'lucide-react'
import { getSamples, uploadFile, uploadSample, runPlan } from '../../api'

const CATEGORY_OPTIONS = [
  { key: 'visualizations',    label: 'Visualizations' },
  { key: 'anomaly_detection', label: 'Anomaly Detection' },
  { key: 'data_quality',      label: 'Quality Audit' },
  { key: 'solutions',         label: 'Solutions' },
]

export default function TriggerPanel({
  readOnly,
  step,
  setStep,
  setSelectedNode,
  sessionId,
  setSessionId,
  setFilename,
  model,
  enabledCategories,
  setEnabledCategories,
  setPlanData,
  setError,
  results,
  filename,
}) {
  const [samples, setSamples] = useState([])
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [statusMsg, setStatusMsg] = useState('')
  const fileInputRef = useRef(null)

  useEffect(() => {
    getSamples()
      .then((data) => setSamples(data.samples || []))
      .catch(() => {})
  }, [])

  const toggleCategory = (key) => {
    setEnabledCategories((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    )
  }

  async function handleUploadAndPlan(uploadFn) {
    setLoading(true)
    setError(null)
    setStatusMsg('Uploading...')
    try {
      const { session_id, filename: fn } = await uploadFn()
      setSessionId(session_id)
      setFilename(fn)
      setStatusMsg('Running planning phase...')
      const { plan } = await runPlan(session_id, model, enabledCategories)
      setPlanData(plan)
      setStep('review')
      setSelectedNode('analysis')
    } catch (e) {
      setError(e.message || 'Failed')
    } finally {
      setLoading(false)
      setStatusMsg('')
    }
  }

  const onFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) handleUploadAndPlan(() => uploadFile(file))
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleUploadAndPlan(() => uploadFile(file))
  }, [model, enabledCategories])

  const onDragOver = (e) => { e.preventDefault(); setDragOver(true) }
  const onDragLeave = () => setDragOver(false)

  // Read-only: show dataset summary
  if (readOnly && results) {
    const schema = results?.tool_results?.find((r) => r.tool === 'infer_schema')?.result || {}
    const shape = schema.shape || {}
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
        <h2 className="text-base font-semibold text-slate-800 mb-1">Dataset</h2>
        <p className="text-sm text-slate-500 mb-4">File: {filename}</p>
        <div className="grid grid-cols-2 gap-3 mb-4">
          <MetricCard label="Rows" value={shape.rows ?? '—'} />
          <MetricCard label="Columns" value={shape.cols ?? '—'} />
        </div>
        {results?.analysis_plan && (
          <details className="mt-2">
            <summary className="text-sm font-semibold text-slate-700 cursor-pointer">Agent's Analysis Plan</summary>
            <p className="text-sm text-slate-600 mt-2 whitespace-pre-wrap leading-relaxed">{results.analysis_plan}</p>
          </details>
        )}
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
      <h2 className="text-base font-semibold text-slate-800 mb-1">Trigger Settings</h2>
      <p className="text-sm text-slate-500 mb-4">Upload a dataset to start the pipeline.</p>

      {/* Drop zone */}
      <div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => fileInputRef.current?.click()}
        className={[
          'border-2 border-dashed rounded-xl p-8 flex flex-col items-center gap-2 cursor-pointer transition-colors',
          dragOver ? 'border-indigo-400 bg-indigo-50' : 'border-slate-300 hover:border-indigo-300 hover:bg-slate-50',
        ].join(' ')}
      >
        <Upload className="w-8 h-8 text-slate-400" />
        <p className="text-sm text-slate-600 font-medium">Drop a CSV here or click to browse</p>
        <p className="text-xs text-slate-400">CSV files supported</p>
        <input ref={fileInputRef} type="file" accept=".csv" className="hidden" onChange={onFileChange} />
      </div>

      {/* Samples */}
      <div className="mt-4">
        <p className="text-xs text-slate-500 mb-2">Or try a sample:</p>
        <div className="flex flex-wrap gap-2">
          {samples.map((s) => (
            <button
              key={s.name}
              disabled={!s.available || loading}
              onClick={() => handleUploadAndPlan(() => uploadSample(s.name))}
              className={[
                'text-xs px-3 py-1.5 rounded-lg border font-medium transition-colors',
                s.available && !loading
                  ? 'border-indigo-200 text-indigo-700 bg-indigo-50 hover:bg-indigo-100'
                  : 'border-slate-200 text-slate-400 cursor-not-allowed',
              ].join(' ')}
            >
              {s.name}
            </button>
          ))}
        </div>
      </div>

      <hr className="my-4 border-slate-200" />

      {/* Analysis options */}
      <div>
        <p className="text-xs font-semibold text-slate-600 mb-2">Analysis Options</p>
        <p className="text-xs text-slate-400 mb-3">Schema and Stats always run. Toggle others to control scope.</p>
        <div className="text-xs text-slate-400 mb-2 pl-1">Schema and Stats — always on</div>
        <div className="grid grid-cols-2 gap-2">
          {CATEGORY_OPTIONS.map((opt) => (
            <label key={opt.key} className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={enabledCategories.includes(opt.key)}
                onChange={() => toggleCategory(opt.key)}
                className="w-4 h-4 rounded accent-indigo-600"
              />
              <span className="text-sm text-slate-700">{opt.label}</span>
            </label>
          ))}
        </div>
      </div>

      {loading && (
        <div className="mt-4 flex items-center gap-2 text-sm text-indigo-600">
          <Spinner />
          <span>{statusMsg || 'Working...'}</span>
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value }) {
  return (
    <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
      <p className="text-xs text-slate-500">{label}</p>
      <p className="text-2xl font-bold text-slate-800 mt-0.5">{value}</p>
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  )
}
