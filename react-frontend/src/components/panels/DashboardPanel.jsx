import React, { useState } from 'react'
import { Activity, BarChart3, TrendingUp, AlertCircle } from 'lucide-react'
import { Card }   from '@/components/ui/card'
import { Badge }  from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { simulateFix } from '@/api'

export default function DashboardPanel({
  results, sessionId, appliedSolutions, simulateResult, setSimulateResult,
}) {
  const [simulating, setSimulating] = useState(false)
  const [simError,   setSimError]   = useState(null)

  if (!results) return <p className="text-sm text-slate-400">Run the pipeline first.</p>

  // Extract quality data
  const qResults = results.quality_tool_results ?? []
  const dqEntry  = qResults.find(r => r.tool === 'compute_data_quality_score')
  const dq       = dqEntry?.result ?? {}
  const score    = dq.overall_score ?? 0
  const issues   = dq.issues ?? []
  const bd       = dq.breakdown ?? {}

  // Extract recommendations count
  const solResults = results.solutions_tool_results ?? []
  const recEntry   = solResults.find(r => r.tool === 'recommend_solutions')
  const recs       = recEntry?.result?.recommendations ?? []
  const totalFixes = recs.reduce((n, r) => n + (r.actions?.length ?? 0), 0)

  const highIssues   = issues.filter(i => i.severity === 'high')
  const mediumIssues = issues.filter(i => i.severity === 'medium')

  const grade = score >= 90 ? 'Excellent' : score >= 75 ? 'Good' : score >= 55 ? 'Fair' : 'Poor'
  const gradeColor = score >= 75 ? 'text-green-600' : score >= 55 ? 'text-yellow-600' : 'text-red-600'

  async function handleSimulate() {
    if (!sessionId) return
    setSimulating(true); setSimError(null)
    try {
      const res = await simulateFix(sessionId, appliedSolutions)
      setSimulateResult(res)
    } catch (e) {
      setSimError(e.message)
    } finally {
      setSimulating(false)
    }
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold flex items-center gap-2">
        <Activity className="h-5 w-5 text-indigo-500" /> Data Health
      </h2>

      {/* Score card */}
      <Card className="p-4">
        <div className="text-sm text-gray-500 mb-1">Health Score</div>
        <div className="flex items-end gap-2">
          <div className="text-3xl font-bold">{score}</div>
          <div className="text-lg text-slate-400 mb-0.5">/ 100</div>
          <div className={`text-sm font-semibold mb-0.5 ml-1 ${gradeColor}`}>{grade}</div>
        </div>
        <div className="mt-3 h-2 rounded-full bg-slate-100 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              score >= 75 ? 'bg-green-500' : score >= 55 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${score}%` }}
          />
        </div>
        <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-slate-500">
          <div>Completeness<br/><span className="font-semibold text-slate-700">{bd.completeness?.toFixed(0) ?? '—'}%</span></div>
          <div>Uniqueness<br/><span className="font-semibold text-slate-700">{bd.uniqueness?.toFixed(0)   ?? '—'}%</span></div>
          <div>Consistency<br/><span className="font-semibold text-slate-700">{bd.consistency?.toFixed(0) ?? '—'}%</span></div>
        </div>
      </Card>

      {/* Issues */}
      {issues.length > 0 ? (
        <div className="space-y-2">
          <p className="text-sm font-medium text-slate-600 flex items-center gap-1.5">
            <AlertCircle className="h-3.5 w-3.5" /> {issues.length} issue{issues.length !== 1 ? 's' : ''} detected
          </p>
          {[...highIssues, ...mediumIssues].slice(0, 6).map((issue, i) => (
            <Card key={i} className="p-3">
              <div className="flex items-start justify-between gap-2">
                <div>
                  <div className="font-medium text-sm text-slate-800">{issue.column}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{issue.issue}</div>
                </div>
                {issue.severity === 'high'
                  ? <Badge className="bg-red-100 text-red-600 border-0 flex-shrink-0">High</Badge>
                  : <Badge className="bg-yellow-100 text-yellow-700 border-0 flex-shrink-0">Medium</Badge>
                }
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card className="p-3">
          <p className="text-sm text-green-700 font-medium">No issues detected — dataset looks clean.</p>
        </Card>
      )}

      {/* Simulate Fix */}
      {totalFixes > 0 && (
        <Card className="p-4">
          <div className="font-medium text-sm mb-1 flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-indigo-500" /> Fix Impact Simulation
          </div>
          <p className="text-xs text-gray-500 mb-3">
            {appliedSolutions.length > 0
              ? `${appliedSolutions.length} fix${appliedSolutions.length > 1 ? 'es' : ''} selected — simulate their impact on the health score`
              : `Apply fixes in the Editor tab, then simulate their impact here`}
          </p>

          {simulateResult && (
            <div className="mb-3 rounded-lg bg-slate-50 p-3 space-y-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-600">
                <TrendingUp className="h-3.5 w-3.5 text-green-600" /> Simulated outcome
              </div>
              <div className="grid grid-cols-3 gap-2 text-center text-xs">
                <div className="rounded-lg bg-white border border-slate-200 p-2">
                  <div className="text-slate-400">Before</div>
                  <div className="text-xl font-bold text-slate-700">{simulateResult.before.overall_score}</div>
                </div>
                <div className="flex items-center justify-center">
                  <div className={`text-lg font-bold ${simulateResult.delta >= 0 ? 'text-green-600' : 'text-red-500'}`}>
                    {simulateResult.delta >= 0 ? '+' : ''}{simulateResult.delta}
                  </div>
                </div>
                <div className="rounded-lg bg-white border border-indigo-200 p-2">
                  <div className="text-slate-400">After</div>
                  <div className={`text-xl font-bold ${simulateResult.delta >= 0 ? 'text-green-600' : 'text-slate-700'}`}>
                    {simulateResult.after.overall_score}
                  </div>
                </div>
              </div>
              <div className="text-xs text-slate-500">
                Issues: {simulateResult.before.issue_count} → {simulateResult.after.issue_count}
                {simulateResult.before.issue_count > simulateResult.after.issue_count && (
                  <span className="ml-1 text-green-600 font-medium">
                    ({simulateResult.before.issue_count - simulateResult.after.issue_count} resolved)
                  </span>
                )}
              </div>
            </div>
          )}

          {simError && <p className="text-xs text-red-500 mb-2">{simError}</p>}

          <Button
            onClick={handleSimulate}
            disabled={simulating || appliedSolutions.length === 0}
            className="w-full bg-indigo-600 hover:bg-indigo-500 text-white"
          >
            {simulating ? 'Simulating...' : 'Simulate Fix'}
          </Button>
          {appliedSolutions.length === 0 && (
            <p className="text-xs text-slate-400 text-center mt-1">
              Apply fixes in the Editor → Suggest Fixes panel first
            </p>
          )}
        </Card>
      )}
    </div>
  )
}
