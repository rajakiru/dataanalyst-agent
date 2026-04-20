// API client for AutoAnalyst backend

const BASE = ''  // proxied by Vite to http://localhost:8000

export async function uploadFile(file) {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${BASE}/api/upload`, { method: 'POST', body: fd })
  if (!res.ok) throw new Error(await res.text())
  return res.json()  // { session_id, filename }
}

export async function uploadSample(name) {
  const res = await fetch(`${BASE}/api/upload-sample`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()  // { session_id, filename }
}

export async function getSamples() {
  const res = await fetch(`${BASE}/api/samples`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()  // { samples: [{name, available}] }
}

export async function runPlan(session_id, model, enabled_categories) {
  const res = await fetch(`${BASE}/api/plan`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id, model, enabled_categories }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()  // { plan: {...}, events: [...] }
}

export function runExecute(session_id, approved_tools, model, onEvent) {
  return new Promise((resolve, reject) => {
    fetch(`${BASE}/api/execute/${session_id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ approved_tools, model }),
    }).then(async (res) => {
      if (!res.ok) {
        const text = await res.text()
        reject(new Error(text))
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      const read = () => {
        reader.read().then(({ done, value }) => {
          if (done) {
            resolve()
            return
          }
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop()  // keep incomplete line

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const event = JSON.parse(line.slice(6))
                onEvent(event)
                if (event.type === 'complete' || event.type === 'error') {
                  resolve(event)
                  return
                }
              } catch (e) {
                // ignore parse errors
              }
            }
          }
          read()
        }).catch(reject)
      }
      read()
    }).catch(reject)
  })
}

export async function getResults(session_id) {
  const res = await fetch(`${BASE}/api/results/${session_id}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function downloadReport(session_id, selected_solutions = []) {
  const res = await fetch(`${BASE}/api/download/report/${session_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ selected_solutions }),
  })
  if (!res.ok) throw new Error(await res.text())
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `report.md`
  a.click()
  URL.revokeObjectURL(url)
}

export async function downloadCleanedCsv(session_id, applied_solutions = []) {
  const res = await fetch(`${BASE}/api/download/cleaned-csv/${session_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ applied_solutions }),
  })
  if (!res.ok) throw new Error(await res.text())
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `cleaned.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export async function downloadPlotsZip(session_id) {
  const res = await fetch(`${BASE}/api/plots/zip/${session_id}`)
  if (!res.ok) throw new Error(await res.text())
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `plots.zip`
  a.click()
  URL.revokeObjectURL(url)
}

export function getPlotUrl(session_id, filename) {
  return `${BASE}/api/plots/${session_id}/${filename}`
}

export async function simulateFix(session_id, applied_solutions = []) {
  const res = await fetch(`${BASE}/api/simulate-fix/${session_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ applied_solutions }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()  // { before: {overall_score, issue_count}, after: {overall_score, issue_count}, delta }
}
