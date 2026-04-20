import React, { useState } from 'react'
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

      <div className="flex h-[calc(100vh-65px)]">
        {/* LEFT: Workflow canvas */}
        <div className="flex-1 bg-slate-100 p-6 overflow-y-auto">
          <Pipeline nodeStates={nodeStates} selectedNode={selectedNode}
                    setSelectedNode={setSelectedNode} step={step} results={results} />
        </div>

        {/* RIGHT: Dynamic panel — fixed 420px */}
        <div className="w-[420px] border-l border-slate-200 bg-white overflow-y-auto flex flex-col">
          <RightPanel {...shared} nodeStates={nodeStates} />
        </div>
      </div>
    </div>
  )
}
