// 現答 대시보드 — 라이트/라임 액센트(레퍼런스: AI-BL 헬스 대시보드 + Vercel).
import { useEffect, useState } from 'react'
import { clsx } from 'clsx'
import { Activity, BarChart3, Asterisk, ChevronDown } from 'lucide-react'
import { useDiagnosisStore } from './shared/store/diagnosis.store'
import type { Category, Gate } from './shared/types'
import { DiagnosisContainer } from './containers/diagnosis/DiagnosisContainer'
import { TrustContainer } from './containers/trust/TrustContainer'

const CATS: Category[] = ['가스', '미성형', '정상']
const GATE_LABEL: Record<Gate, string> = {
  supervised: '지도 GBM',
  mahalanobis: '비지도 Mahalanobis',
}

const s = {
  page: 'min-h-screen flex justify-center p-4 sm:p-6',
  surface: 'w-full max-w-[1180px] bg-white rounded-[28px] shadow-card flex overflow-hidden',
  rail: 'w-16 shrink-0 border-r border-zinc-100 flex flex-col items-center gap-3 py-5',
  brand: 'w-9 h-9 rounded-full bg-ink text-accent grid place-items-center mb-2',
  railBtn: 'w-10 h-10 rounded-full grid place-items-center transition',
  railOn: 'bg-accent text-ink',
  railOff: 'text-zinc-400 hover:bg-zinc-100',
  main: 'flex-1 p-7 min-w-0',
  topbar: 'flex items-center gap-3 mb-7',
  gatePill: 'flex items-center gap-2 rounded-full border border-zinc-200 px-4 py-2 text-sm font-medium cursor-pointer hover:border-zinc-300',
  title: 'text-4xl font-extrabold tracking-tight',
  subtitle: 'mt-2 text-sm text-zinc-400',
  controls: 'flex flex-wrap items-center gap-3 mt-6 mb-6',
  chip: 'rounded-full px-4 py-2 text-sm font-medium transition',
  chipOn: 'bg-ink text-white',
  chipOff: 'bg-zinc-100 text-zinc-500 hover:bg-zinc-200',
  slider: 'flex-1 min-w-[160px] accent-ink',
  truth: 'rounded-full bg-accent px-3 py-1.5 text-sm font-semibold text-ink',
}

export default function App() {
  const { gate, cat, shots, pos, init, setGate, setCat, setPos } = useDiagnosisStore()
  const [tab, setTab] = useState<'diagnosis' | 'trust'>('diagnosis')

  useEffect(() => {
    void init()
  }, [init])

  const toggleGate = () => void setGate(gate === 'supervised' ? 'mahalanobis' : 'supervised')
  const current = shots[pos]

  return (
    <div className={s.page}>
      <div className={s.surface}>
        <nav className={s.rail}>
          <span className={s.brand}><Asterisk size={20} /></span>
          <button type="button" title="실시간 진단" onClick={() => setTab('diagnosis')}
            className={clsx(s.railBtn, tab === 'diagnosis' ? s.railOn : s.railOff)}>
            <Activity size={20} />
          </button>
          <button type="button" title="측정된 신뢰" onClick={() => setTab('trust')}
            className={clsx(s.railBtn, tab === 'trust' ? s.railOn : s.railOff)}>
            <BarChart3 size={20} />
          </button>
        </nav>

        <main className={s.main}>
          <div className={s.topbar}>
            <button type="button" className={s.gatePill} onClick={toggleGate}>
              gate: <span className="font-semibold">{GATE_LABEL[gate]}</span>
              <ChevronDown size={15} className="text-zinc-400" />
            </button>
            <span className="text-sm text-zinc-400">CN7 · 우진2호기</span>
          </div>

          <h1 className={s.title}>사출성형 결함진단</h1>
          <p className={s.subtitle}>한 shot이 게이트 → 근거 → 모드 → 처방을 흐르는 실시간 진단</p>

          <div className={s.controls}>
            {CATS.map((c) => (
              <button key={c} type="button" onClick={() => void setCat(c)}
                className={clsx(s.chip, cat === c ? s.chipOn : s.chipOff)}>
                {c}
              </button>
            ))}
            <input type="range" min={0} max={Math.max(shots.length - 1, 0)} value={pos}
              onChange={(e) => void setPos(Number(e.target.value))} className={s.slider} />
            {current && <span className={s.truth}>실제 {current.groundTruth}</span>}
          </div>

          {tab === 'diagnosis' ? <DiagnosisContainer /> : <TrustContainer />}
        </main>
      </div>
    </div>
  )
}
