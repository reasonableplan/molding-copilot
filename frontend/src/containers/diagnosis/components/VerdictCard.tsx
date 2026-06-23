// 판정 카드 — 이상=강한 다크 카드(레드 닷+모드 라임), 모른다=라이트 카드.
import { cva } from 'class-variance-authority'
import type { Diagnosis } from '../../../shared/types'

const card = cva('rounded-card px-6 py-5 flex items-center gap-4', {
  variants: {
    tone: {
      anomaly: 'bg-ink text-white',
      unknown: 'bg-zinc-50 border border-zinc-200 text-zinc-700',
    },
  },
})

export function VerdictCard({ diagnosis }: { diagnosis: Diagnosis }) {
  const anomaly = diagnosis.isAnomaly
  return (
    <div className={card({ tone: anomaly ? 'anomaly' : 'unknown' })}>
      <span className={anomaly ? 'w-3 h-3 rounded-full bg-gas shadow-[0_0_10px_#ef4444]' : 'w-3 h-3 rounded-full bg-zinc-300'} />
      {anomaly ? (
        <div className="flex items-baseline gap-3">
          <span className="text-xl font-bold">이상 감지</span>
          {diagnosis.mode ? (
            <span className="text-xl font-bold text-accent">{diagnosis.mode}</span>
          ) : (
            <span className="text-sm text-zinc-400">모드 미상 · 단일변수 근거 약함(다변량 패턴)</span>
          )}
        </div>
      ) : (
        <div>
          <span className="text-xl font-bold">모른다 · 근거 불충분</span>
          <p className="text-sm text-zinc-400 mt-0.5">정상과 구분되지 않음 → 원인을 단정하지 않음(지어내지 않음).</p>
        </div>
      )}
      <span className="ml-auto text-sm tnum text-zinc-400">gate {diagnosis.gateMode}</span>
    </div>
  )
}
