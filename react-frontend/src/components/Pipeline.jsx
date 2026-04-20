import React from 'react'
import { motion } from 'framer-motion'
import { Database, Settings, AlertTriangle, Wand2, FileText, Plus } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'

const NODES = [
  { id: 'trigger',   Icon: Database,      title: 'Dataset Uploaded', defaultDesc: 'Trigger when CSV uploaded'   },
  { id: 'analysis',  Icon: Settings,      title: 'Run Analysis',     defaultDesc: 'Schema + stats + anomalies'  },
  { id: 'quality',   Icon: AlertTriangle, title: 'Detect Issues',    defaultDesc: 'Find data cascade risks'     },
  { id: 'solutions', Icon: Wand2,         title: 'Suggest Fixes',    defaultDesc: 'Impute, transform, clean'    },
  { id: 'report',    Icon: FileText,      title: 'Generate Report',  defaultDesc: 'Export findings'             },
]

function getDynamicDesc(nodeId, results) {
  if (!results) return null
  const byTool = {}
  ;(results.tool_results ?? []).forEach(r => { byTool[r.tool] = r })

  switch (nodeId) {
    case 'trigger': {
      const shape = byTool['infer_schema']?.result?.shape
      return shape ? `${shape.rows} rows, ${shape.cols} columns` : null
    }
    case 'analysis': {
      const n = (results.tool_results ?? []).length
      return n ? `${n} analyses completed` : null
    }
    case 'quality': {
      const q = (results.quality_tool_results ?? []).find(r => r.tool === 'compute_data_quality_score')
      const issues = q?.result?.issues ?? []
      const score  = q?.result?.overall_score
      if (score !== undefined) return `Score ${score}/100 · ${issues.length} issue${issues.length !== 1 ? 's' : ''}`
      return null
    }
    case 'solutions': {
      const st = (results.solutions_tool_results ?? []).find(r => r.tool === 'recommend_solutions')
      const n  = st?.result?.recommendations?.length ?? 0
      return n ? `${n} fix${n !== 1 ? 'es' : ''} recommended` : null
    }
    case 'report':
      return results.narrative ? 'Report ready to export' : null
  }
  return null
}

function getStatusBadge(state) {
  if (state === 'complete') return <Badge className="bg-green-100 text-green-700 border-0">Done</Badge>
  if (state === 'warning')  return <Badge className="bg-yellow-100 text-yellow-700 border-0">Attention</Badge>
  if (state === 'running')  return <Badge className="bg-indigo-100 text-indigo-700 border-0 animate-pulse">Running</Badge>
  return null
}

export default function Pipeline({ nodeStates, selectedNode, setSelectedNode, step, results }) {
  function isClickable(nodeId, state) {
    if (step === 'complete') return true
    if (state === 'idle' && nodeId !== 'trigger') return false
    return true
  }

  return (
    <div className="mx-auto max-w-2xl space-y-0">
      {NODES.map((node, i) => {
        const { id, Icon, title, defaultDesc } = node
        const state    = nodeStates[id] || 'idle'
        const isSel    = selectedNode === id
        const clickable = isClickable(id, state)
        const desc     = getDynamicDesc(id, results) || defaultDesc

        const cardCls = [
          'cursor-pointer rounded-xl border bg-white p-4 shadow-sm transition-all duration-150',
          isSel ? 'border-indigo-500' : 'border-gray-200',
          clickable && !isSel ? 'hover:border-indigo-300 hover:shadow-md' : '',
          !clickable ? 'opacity-50 cursor-default' : '',
        ].filter(Boolean).join(' ')

        return (
          <React.Fragment key={id}>
            <motion.div
              whileHover={clickable ? { scale: 1.015 } : {}}
              onClick={() => clickable && setSelectedNode(id)}
              className={cardCls}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: clickable ? 1 : 0.5, y: 0 }}
              transition={{ delay: i * 0.06 }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Icon className="h-5 w-5 text-indigo-500 flex-shrink-0" />
                  <div>
                    <div className="font-medium text-slate-900">{title}</div>
                    <div className="text-sm text-gray-500">{desc}</div>
                  </div>
                </div>
                {getStatusBadge(state)}
              </div>
            </motion.div>

            {i < NODES.length - 1 && (
              <div className="flex justify-center py-1">
                <div className="h-5 w-px bg-gray-300" />
              </div>
            )}
          </React.Fragment>
        )
      })}

      <div className="flex justify-center pt-3">
        <Button variant="ghost" className="gap-2 text-slate-400">
          <Plus className="h-4 w-4" /> Add Step
        </Button>
      </div>
    </div>
  )
}
