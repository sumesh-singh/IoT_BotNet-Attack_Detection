import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getJson, postJson } from './client'

export function useDatasets() {
  return useQuery({ queryKey: ['datasets'], queryFn: () => getJson<{ items: any[] }>('datasets') })
}

export function useValidateDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => postJson<any>('datasets/validate', { name }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ['datasets'] }) }
  })
}
