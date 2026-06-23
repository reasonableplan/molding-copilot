// 변수별 z-편차 — 라운드 막대, 0 기준 좌우, 시그니처 색 + 값 pill.
import { clsx } from 'clsx'
import type { ZBar } from '../types'

const SCALE = 4 // ±4σ = 막대 절반

const s = {
  wrap: 'rounded-card bg-white shadow-card p-6',
  title: 'text-xs font-medium uppercase tracking-wide text-zinc-400',
  legend: 'text-[11px] text-zinc-400 mb-4',
  row: 'flex items-center gap-3 py-1',
  name: 'w-40 shrink-0 text-right text-xs text-zinc-500 truncate',
  lane: 'relative flex-1 h-6',
  center: 'absolute inset-y-0 left-1/2 w-px bg-zinc-200',
  bar: 'absolute inset-y-1 rounded-full',
  val: 'absolute top-1/2 -translate-y-1/2 text-[11px] font-semibold tnum',
}

function barColor(mode: string | null): string {
  if (mode === '가스') return 'bg-gas'
  if (mode === '미성형') return 'bg-short'
  return 'bg-zinc-300'
}
function valColor(mode: string | null): string {
  if (mode === '가스') return 'text-gas'
  if (mode === '미성형') return 'text-short'
  return 'text-zinc-400'
}

export function ZBars({ bars }: { bars: ZBar[] }) {
  return (
    <div className={s.wrap}>
      <div className={s.title}>변수별 이탈 · z-편차</div>
      <div className={s.legend}>빨강=과열·가스 · 파랑=충전·미성형 · 회색=기타 · 중앙선=0σ</div>
      {bars.map((b) => {
        const half = Math.min(Math.abs(b.z) / SCALE, 1) * 50
        const pos = b.z >= 0
        return (
          <div key={b.var} className={s.row}>
            <span className={s.name}>{b.var}</span>
            <div className={s.lane}>
              <div className={s.center} />
              <div className={clsx(s.bar, barColor(b.mode))}
                style={pos ? { left: '50%', width: `${half}%` } : { right: '50%', width: `${half}%` }} />
              <span className={clsx(s.val, valColor(b.mode))}
                style={pos ? { left: `calc(50% + ${half}% + 6px)` } : { right: `calc(50% + ${half}% + 6px)` }}>
                {b.z > 0 ? '+' : ''}{b.z}σ
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
