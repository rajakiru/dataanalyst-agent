import { Play } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

export default function TopBar({ model, setModel, models, step, reset, mode, setMode }) {
  return (
    <div className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4 h-[65px]">
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-semibold tracking-tight">AutoAnalyst</h1>
        <Badge variant="secondary">Workflow</Badge>
      </div>

      <div className="flex items-center gap-2">
        <select
          value={model}
          onChange={e => setModel(e.target.value)}
          className="h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-700 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          {models.map(m => <option key={m}>{m}</option>)}
        </select>

        <Button
          variant="outline"
          className={mode === 'editor' ? 'border-indigo-400 text-indigo-600 bg-indigo-50' : ''}
          onClick={() => setMode('editor')}
        >
          Editor
        </Button>
        <Button
          variant="outline"
          className={mode === 'dashboard' ? 'border-indigo-400 text-indigo-600 bg-indigo-50' : ''}
          onClick={() => setMode('dashboard')}
        >
          Dashboard
        </Button>

        {step === 'complete' ? (
          <Button variant="outline" onClick={reset}>New Analysis</Button>
        ) : (
          <Button className="bg-indigo-600 hover:bg-indigo-500 text-white">
            <Play className="mr-1.5 h-3.5 w-3.5" /> Run
          </Button>
        )}
      </div>
    </div>
  )
}
