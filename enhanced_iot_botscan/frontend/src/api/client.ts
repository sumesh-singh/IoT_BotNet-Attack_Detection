import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE_URL || '/api/v1'
export const api = axios.create({ baseURL })

export async function getJson<T>(url: string) {
  const res = await api.get<T>(url)
  return res.data
}

export async function postJson<T>(url: string, data: unknown) {
  const res = await api.post<T>(url, data)
  return res.data
}
