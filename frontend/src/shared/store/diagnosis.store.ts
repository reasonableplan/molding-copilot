// Zustand 단일 store — action 에서 axios 직접 호출(react-vite 프로파일: React Query 금지).
import { create } from 'zustand'
import { fetchDiagnosis, fetchShots, fetchTrust } from '../api/client'
import type { Category, Diagnosis, Gate, ShotItem, Trust } from '../types'

interface DiagnosisState {
  gate: Gate
  cat: Category
  shots: ShotItem[]
  pos: number // shots 내 위치
  diagnosis: Diagnosis | null
  trust: Trust | null
  loading: boolean
  error: string | null
  init: () => Promise<void>
  setGate: (gate: Gate) => Promise<void>
  setCat: (cat: Category) => Promise<void>
  setPos: (pos: number) => Promise<void>
}

export const useDiagnosisStore = create<DiagnosisState>((set, get) => ({
  gate: 'supervised',
  cat: '가스',
  shots: [],
  pos: 0,
  diagnosis: null,
  trust: null,
  loading: false,
  error: null,

  init: async () => {
    const trust = await fetchTrust()
    set({ trust })
    await get().setCat('가스')
  },

  setGate: async (gate) => {
    set({ gate })
    await get().setPos(get().pos)
  },

  setCat: async (cat) => {
    set({ loading: true, error: null })
    try {
      const shots = await fetchShots(cat)
      set({ cat, shots, pos: 0 })
      const diagnosis = await fetchDiagnosis(shots[0].idx, get().gate)
      set({ diagnosis, loading: false })
    } catch {
      set({ error: '데이터를 불러오지 못했습니다', loading: false })
    }
  },

  setPos: async (pos) => {
    const { shots, gate } = get()
    if (!shots[pos]) return
    set({ loading: true, error: null, pos })
    try {
      const diagnosis = await fetchDiagnosis(shots[pos].idx, gate)
      set({ diagnosis, loading: false })
    } catch {
      set({ error: '진단을 불러오지 못했습니다', loading: false })
    }
  },
}))
