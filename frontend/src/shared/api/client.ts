// 단일 axios 인스턴스. dev 는 vite 프록시로 /api → Django(8000).
import axios from 'axios'
import type { Category, Diagnosis, Gate, ShotItem, Trust } from '../types'

const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL ?? '/api',
})

export async function fetchShots(cat: Category): Promise<ShotItem[]> {
  const { data } = await client.get<ShotItem[]>('/shots', { params: { cat } })
  return data
}

export async function fetchDiagnosis(idx: number, gate: Gate): Promise<Diagnosis> {
  const { data } = await client.get<Diagnosis>('/diagnose', { params: { idx, gate } })
  return data
}

export async function fetchTrust(): Promise<Trust> {
  const { data } = await client.get<Trust>('/trust')
  return data
}
