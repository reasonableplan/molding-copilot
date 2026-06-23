// conformal p-value 미터 — 큰 굵은 숫자 + pill 배지 + 라운드 미터(레드존+마커).
import { clsx } from 'clsx'

interface GaugeProps {
  pValue: number
  alpha: number
  anomaly: boolean
}

const s = {
  wrap: 'rounded-card bg-white shadow-card p-6 flex flex-col',
  label: 'text-xs font-medium uppercase tracking-wide text-zinc-400',
  row: 'flex items-end justify-between mt-2',
  value: 'text-5xl font-extrabold tnum leading-none',
  badge: 'rounded-full px-3 py-1 text-xs font-semibold',
  track: 'relative mt-6 h-3 rounded-full bg-zinc-100 overflow-hidden',
  redzone: 'absolute inset-y-0 left-0 bg-gas/25',
  marker: 'absolute inset-y-[-3px] w-1 rounded bg-ink',
  ticks: 'mt-2 flex justify-between text-[11px] tnum text-zinc-400',
  note: 'mt-auto pt-5 text-sm text-zinc-500 leading-relaxed border-t border-zinc-100',
  noteKey: 'font-semibold text-ink',
}

export function Gauge({ pValue, alpha, anomaly }: GaugeProps) {
  return (
    <div className={s.wrap}>
      <div className={s.label}>conformal p-value</div>
      <div className={s.row}>
        <span className={clsx(s.value, anomaly ? 'text-gas' : 'text-ink')}>{pValue.toFixed(3)}</span>
        <span className={clsx(s.badge, anomaly ? 'bg-gas/10 text-gas' : 'bg-zinc-100 text-zinc-500')}>
          {anomaly ? '이상' : '모른다'} · α={alpha}
        </span>
      </div>
      <div className={s.track}>
        <div className={s.redzone} style={{ width: `${alpha * 100}%` }} />
        <div className={s.marker} style={{ left: `${Math.min(pValue, 1) * 100}%` }} />
      </div>
      <div className={s.ticks}><span>0</span><span>α={alpha}</span><span>1</span></div>
      <p className={s.note}>
        {anomaly ? (
          <>관측 p가 보정 임계 <span className={s.noteKey}>α={alpha}</span> 이하 → conformal 게이트 발화 = <span className={s.noteKey}>이상 판정</span>. 오탐율 ≤ α 유한표본 보장.</>
        ) : (
          <>관측 p가 <span className={s.noteKey}>α={alpha}</span> 초과 → 게이트 닫힘 = <span className={s.noteKey}>기권(모른다)</span>. LLM을 거치지 않음.</>
        )}
      </p>
    </div>
  )
}
