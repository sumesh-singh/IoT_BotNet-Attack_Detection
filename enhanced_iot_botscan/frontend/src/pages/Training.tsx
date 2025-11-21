import { useState } from 'react'
import { useDatasets } from '../api/datasets'
import { useCreateRun, useRun, useRunResults } from '../api/training'

export default function Training() {
  const { data: ds } = useDatasets()
  const [dataset, setDataset] = useState<string>('n_baiot')
  const [mode, setMode] = useState<'baseline'|'adversarial'|'full'>('baseline')
  const createRun = useCreateRun()
  const runId = (createRun.data as any)?.id || null
  const run = useRun(runId)
  const results = useRunResults(runId)
  const items = ds?.items || []
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Training</h1>
      <div className="bg-white p-4 rounded shadow space-y-3">
        <div>
          <label className="block text-sm mb-1">Dataset</label>
          <select className="border rounded px-2 py-1" value={dataset} onChange={e => setDataset(e.target.value)}>
            {items.map((d: any) => (
              <option key={d.name} value={d.name}>{d.name}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm mb-1">Mode</label>
          <select className="border rounded px-2 py-1" value={mode} onChange={e => setMode(e.target.value as any)}>
            <option value="baseline">baseline</option>
            <option value="adversarial">adversarial</option>
            <option value="full">full</option>
          </select>
        </div>
        <button
          className="px-3 py-1 bg-blue-600 text-white rounded"
          onClick={() => createRun.mutate({ datasets: [dataset], mode })}
          disabled={createRun.isLoading}
        >Start Run</button>
        {runId && (
          <div className="mt-4">
            <div>Run ID: {runId}</div>
            <div>Status: {run.data?.status || 'queued'}</div>
          </div>
        )}
        {results.isSuccess && (
          <div className="bg-green-100 border border-green-300 p-3 rounded mt-2">
            Results ready
          </div>
        )}
      </div>
    </div>
  )
}
