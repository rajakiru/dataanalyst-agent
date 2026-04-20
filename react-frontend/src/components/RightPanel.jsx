import React from 'react'
import TriggerPanel   from './panels/TriggerPanel'
import ReviewPanel    from './panels/ReviewPanel'
import RunningPanel   from './panels/RunningPanel'
import AnalysisPanel  from './panels/AnalysisPanel'
import QualityPanel   from './panels/QualityPanel'
import SolutionsPanel from './panels/SolutionsPanel'
import ReportPanel    from './panels/ReportPanel'
import DashboardPanel from './panels/DashboardPanel'

export default function RightPanel(props) {
  const { step, mode, selectedNode } = props

  if (step === 'upload')   return <div className="p-6 flex-1"><TriggerPanel {...props} /></div>
  if (step === 'review')   return <div className="p-6 flex-1"><ReviewPanel  {...props} /></div>
  if (step === 'running')  return <div className="p-6 flex-1"><RunningPanel {...props} /></div>

  // complete
  if (step === 'complete') {
    // Dashboard mode always shows the aggregate health dashboard
    if (mode === 'dashboard') return <div className="p-6 flex-1"><DashboardPanel {...props} /></div>

    // Editor mode routes by selected node
    return (
      <div className="p-6 flex-1">
        {selectedNode === 'trigger'   && <TriggerPanel   {...props} readOnly />}
        {selectedNode === 'analysis'  && <AnalysisPanel  {...props} />}
        {selectedNode === 'quality'   && <QualityPanel   {...props} />}
        {selectedNode === 'solutions' && <SolutionsPanel {...props} />}
        {selectedNode === 'report'    && <ReportPanel    {...props} />}
      </div>
    )
  }

  return null
}
