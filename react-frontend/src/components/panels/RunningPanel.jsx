import React, { useEffect, useRef } from 'react'

const PHASE_LABELS = {
  mcp_init:          'MCP server started',
  schema:            'Inferring schema',
  planning:          'Planning analysis',
  collection:        'Running collection agent',
  quality_score:     'Quality audit — scoring',
  quality_tools:     'Quality audit — checks',
  quality_narrative: 'Quality agent writing report',
  solutions:         'Solutions agent',
  reporting:         'Reporting agent',
}

function fmtTool(tool, args = {}) {
  const col = args.column || args.numeric_column || args.x_column
  return col ? `${tool}(${col})` : tool
}

function fmtResult(tool, result = {}) {
  if (result.error) return `error: ${String(result.error).slice(0, 60)}`
  if (tool === 'infer_schema') {
    const s = result.shape || {}
    return `${s.rows ?? '?'} rows x ${s.cols ?? '?'} cols`
  }
  if (tool === 'summarize_statistics') return `stats for ${Object.keys(result.statistics || {}).length} column(s)`
  if (tool === 'compute_correlations') {
    const top = (result.top_correlations || [])[0]
    return top ? `top: ${top.col1}/${top.col2} r=${top.correlation}` : 'no correlations'
  }
  if (tool === 'detect_anomalies') return `${result.outlier_count ?? 0} outliers (${result.outlier_pct ?? 0}%)`
  if (tool === 'compute_data_quality_score') return `score: ${result.overall_score ?? '?'} / 100`
  if (tool === 'detect_duplicates') return `${result.duplicate_rows ?? 0} duplicate rows`
  if (result.plot_path) return `saved ${result.plot_path.split('/').pop()}`
  return 'done'
}

export default function RunningPanel({ events, step }) {
  const bottomRef = useRef(null)
  const isComplete = step === 'complete'

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [events])

  // Build a structured timeline from events
  const timeline = []
  let currentPhase = null
  let phaseToolCount = 0

  for (const ev of events) {
    if (ev.type === 'phase_start') {
      if (currentPhase) {
        timeline.push({ kind: 'phase_close', phase: currentPhase, count: phaseToolCount })
      }
      currentPhase = ev.phase
      phaseToolCount = 0
      timeline.push({ kind: 'phase_open', phase: ev.phase })
    } else if (ev.type === 'tool_start') {
      timeline.push({ kind: 'tool_running', tool: ev.tool, args: ev.args || {} })
    } else if (ev.type === 'tool_result') {
      // Replace last tool_running entry for same tool with tool_done
      for (let i = timeline.length - 1; i >= 0; i--) {
        if (timeline[i].kind === 'tool_running' && timeline[i].tool === ev.tool) {
          timeline[i] = { kind: 'tool_done', tool: ev.tool, args: timeline[i].args, result: ev.result }
          phaseToolCount++
          break
        }
      }
    } else if (ev.type === 'complete' || ev.type === 'error') {
      if (currentPhase) {
        timeline.push({ kind: 'phase_close', phase: currentPhase, count: phaseToolCount })
      }
    }
  }
  // If still running, show open phase
  if (!isComplete && currentPhase) {
    // Already shown as phase_open
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-sm">
      <div className="flex items-center gap-3 mb-4">
        {!isComplete && <Spinner />}
        <h2 className="text-base font-semibold text-slate-800">
          {isComplete ? 'Analysis Complete' : 'Running Analysis...'}
        </h2>
      </div>

      <div className="space-y-1 max-h-[60vh] overflow-y-auto pr-1">
        {timeline.map((item, i) => {
          if (item.kind === 'phase_open') {
            return (
              <div key={i} className="text-sm font-bold text-slate-700 mt-2 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-indigo-400 inline-block" />
                {PHASE_LABELS[item.phase] || item.phase}...
              </div>
            )
          }
          if (item.kind === 'phase_close') {
            return (
              <div key={i} className="text-sm font-bold text-slate-700 mt-2 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
                {PHASE_LABELS[item.phase] || item.phase}
                {item.count > 0 && <span className="font-normal text-slate-400">— done ({item.count} tools)</span>}
                {item.count === 0 && <span className="font-normal text-slate-400">— done</span>}
              </div>
            )
          }
          if (item.kind === 'tool_running') {
            return (
              <div key={i} className="ml-6 flex items-center gap-2 text-xs text-slate-500">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-400 inline-block animate-pulse" />
                <code className="bg-slate-100 px-1.5 py-0.5 rounded">{fmtTool(item.tool, item.args)}</code>
              </div>
            )
          }
          if (item.kind === 'tool_done') {
            return (
              <div key={i} className="ml-6 flex items-center gap-2 text-xs text-slate-500">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
                <code className="bg-slate-100 px-1.5 py-0.5 rounded">{fmtTool(item.tool, item.args)}</code>
                <span className="text-slate-400">— {fmtResult(item.tool, item.result)}</span>
              </div>
            )
          }
          return null
        })}
        <div ref={bottomRef} />
      </div>

      {isComplete && (
        <div className="mt-4 p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-sm text-emerald-700 font-medium">
          Analysis complete. Select a node in the pipeline to explore results.
        </div>
      )}
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin w-5 h-5 text-indigo-500" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
    </svg>
  )
}
