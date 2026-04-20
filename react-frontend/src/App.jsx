import React, { useState, useCallback, useRef } from 'react'
import TopBar from './components/TopBar'
import Pipeline from './components/Pipeline'
import RightPanel from './components/RightPanel'

const AVAILABLE_MODELS = ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1']

export default function App() {
  const [step, setStep]               = useState('upload')   // upload|review|running|complete
  const [mode, setMode]               = useState('editor')   // editor|dashboard
  const [selectedNode, setSelectedNode] = useState('trigger')

  const [sessionId, setSessionId]     = useState(null)
  const [filename, setFilename]       = useState('')
  const [model, setModel]             = useState(AVAILABLE_MODELS[0])
  const [enabledCategories, setEnabledCategories] = useState([
    'visualizations', 'anomaly_detection', 'data_quality', 'solutions',
  ])

  const [planData, setPlanData]       = useState(null)
  const [approvedTools, setApprovedTools] = useState([])
  const [events, setEvents]           = useState([])
  const [results, setResults]         = useState(null)
  const [error, setError]             = useState(null)

  // Dismissed issues + selected/applied solutions (lifted so both panels share them)
  const [dismissedIssues, setDismissedIssues]       = useState(new Set())
  const [selectedSolutions, setSelectedSolutions]   = useState(new Set())
  const [appliedSolutions, setAppliedSolutions]     = useState([])
  const [simulateResult, setSimulateResult]         = useState(null) // {before, after, delta}

  function getNodeStates() {
    if (step === 'upload')   return { trigger:'idle',     analysis:'idle',    quality:'idle',    solutions:'idle',    report:'idle'    }
    if (step === 'review')   return { trigger:'complete', analysis:'idle',    quality:'idle',    solutions:'idle',    report:'idle'    }
    if (step === 'running')  return { trigger:'complete', analysis:'running', quality:'idle',    solutions:'idle',    report:'idle'    }
    // complete — quality node is 'warning' if there are active issues
    const hasIssues = (() => {
      if (!results) return false
      const q = results.quality_tool_results ?? []
      const dq = q.find(r => r.tool === 'compute_data_quality_score')?.result
      return (dq?.issues?.length ?? 0) > 0
    })()
    return {
      trigger:   'complete',
      analysis:  'complete',
      quality:   hasIssues ? 'warning' : 'complete',
      solutions: 'complete',
      report:    'complete',
    }
  }

  function reset() {
    setStep('upload'); setMode('editor'); setSelectedNode('trigger')
    setSessionId(null); setFilename(''); setPlanData(null)
    setApprovedTools([]); setEvents([]); setResults(null); setError(null)
    setDismissedIssues(new Set()); setSelectedSolutions(new Set())
    setAppliedSolutions([]); setSimulateResult(null)
  }

  const nodeStates = getNodeStates()

  // Resizable split
  const [leftWidth, setLeftWidth] = useState(null) // null = flex-1 default
  const dragging = useRef(false)
  const startX = useRef(0)
  const startW = useRef(0)
  const containerRef = useRef(null)

  const onMouseDown = useCallback((e) => {
    dragging.current = true
    startX.current = e.clientX
    startW.current = containerRef.current
      ? containerRef.current.getBoundingClientRect().width *
        (leftWidth != null ? leftWidth / 100 : 0.58)
      : e.clientX
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [leftWidth])

  const onMouseMove = useCallback((e) => {
    if (!dragging.current || !containerRef.current) return
    const totalW = containerRef.current.getBoundingClientRect().width
    const newW = startW.current + (e.clientX - startX.current)
    const pct = Math.min(75, Math.max(25, (newW / totalW) * 100))
    setLeftWidth(pct)
  }, [])

  const onMouseUp = useCallback(() => {
    dragging.current = false
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }, [])

  const shared = {
    step, setStep, mode, setMode,
    selectedNode, setSelectedNode,
    sessionId, setSessionId, filename, setFilename,
    model, setModel,
    enabledCategories, setEnabledCategories,
    planData, setPlanData,
    approvedTools, setApprovedTools,
    events, setEvents,
    results, setResults,
    error, setError,
    dismissedIssues, setDismissedIssues,
    selectedSolutions, setSelectedSolutions,
    appliedSolutions, setAppliedSolutions,
    simulateResult, setSimulateResult,
    reset,
    availableModels: AVAILABLE_MODELS,
  }

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <TopBar model={model} setModel={setModel} models={AVAILABLE_MODELS}
              step={step} reset={reset} mode={mode} setMode={setMode} />

      <div
        ref={containerRef}
        className="flex h-[calc(100vh-65px)]"
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseUp}
      >
        {/* LEFT: Workflow canvas */}
        <div
          className="bg-slate-100 p-6 overflow-y-auto"
          style={leftWidth != null ? { width: `${leftWidth}%` } : { flex: 1 }}
        >
          <Pipeline nodeStates={nodeStates} selectedNode={selectedNode}
                    setSelectedNode={setSelectedNode} step={step} results={results} />
        </div>

        {/* Drag handle */}
        <div
          onMouseDown={onMouseDown}
          className="w-1.5 bg-slate-200 hover:bg-indigo-400 cursor-col-resize transition-colors flex-shrink-0"
          title="Drag to resize"
        />

        {/* RIGHT: Dynamic panel */}
        <div
          className="border-slate-200 bg-white overflow-y-auto flex flex-col"
          style={leftWidth != null ? { width: `${100 - leftWidth}%` } : { width: '420px' }}
        >
          <RightPanel {...shared} nodeStates={nodeStates} />
        </div>
      </div>
    </div>
  )
}
