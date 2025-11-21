import { useQuery } from '@tanstack/react-query'
import { getJson } from './client'

export function useModels() {
  return useQuery({ queryKey: ['models'], queryFn: () => getJson<{ items: any[] }>('models') })
}

export function useReports() {
  return useQuery({ queryKey: ['reports'], queryFn: () => getJson<{ items: any[] }>('reports') })
}
