import React, { useState } from 'react'
import { downloadReport, downloadPlotsZip } from '../../api'

export default function ReportPanel({ results, sessionId, filename }) {
  const [selectedSolutions, setSelectedSolutions] = useState([])
  const narrative = results?.narrative || ''
  const plotPaths = [...(results?.plot_paths || []), ...(results?.quality_plot_paths || [])]

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
      <div className="p-5 space-y-5">
        <h2 className="text-base font-semibold text-slate-800">Generated Report</h2>

        <div className="flex gap-3">
          <button
            onClick={() => downloadReport(sessionId, selectedSolutions)}
            className="text-sm bg-indigo-600 text-white px-4 py-2 rounded-lg font-semibold hover:bg-indigo-700 transition-colors"
          >
            Download Report (.md)
          </button>
          {plotPaths.length > 0 && (
            <button
              onClick={() => downloadPlotsZip(sessionId)}
              className="text-sm border border-slate-200 text-slate-700 px-4 py-2 rounded-lg font-semibold hover:bg-slate-50 transition-colors"
            >
              Download Plots (.zip)
            </button>
          )}
        </div>

        <hr className="border-slate-200" />

        {narrative ? (
          <div>
            <h3 className="text-sm font-semibold text-slate-700 mb-3">Narrative Report</h3>
            <pre className="text-sm text-slate-700 whitespace-pre-wrap leading-relaxed font-sans bg-slate-50 rounded-xl border border-slate-200 p-4 overflow-y-auto max-h-[60vh]">
              {narrative}
            </pre>
          </div>
        ) : (
          <p className="text-sm text-slate-500">No narrative available.</p>
        )}
      </div>
    </div>
  )
}
