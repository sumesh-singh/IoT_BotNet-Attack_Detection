import { useMutation, useQuery } from '@tanstack/react-query'
import { getJson, postJson } from './client'

export type CreateRunPayload = { datasets: string[], mode: 'baseline'|'adversarial'|'full' }

export function useCreateRun() {
  return useMutation({ mutationFn: (payload: CreateRunPayload) => postJson<{ id: string }>('training/runs', payload) })
}

export function useRun(runId: string | null) {
  return useQuery({
    queryKey: ['run', runId],
    queryFn: () => getJson<any>(`training/runs/${runId}`),
    enabled: !!runId,
    refetchInterval: (q) => {
      const status = q.state.data?.status
      return status && status === 'running' ? 2000 : false
    }
  })
}

export function useRunResults(runId: string | null) {
  return useQuery({
    queryKey: ['run-results', runId],
    queryFn: () => getJson<any>(`training/runs/${runId}/results`),
    enabled: !!runId
  })
}
