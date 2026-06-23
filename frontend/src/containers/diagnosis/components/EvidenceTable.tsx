// 근거 테이블 — 실측 인용(LLM 날조 아님).
import type { EvidenceItem } from '../../../shared/types'

const s = {
  wrap: 'rounded-card bg-white shadow-card p-6',
  title: 'text-xs font-medium uppercase tracking-wide text-zinc-400 mb-4',
  table: 'w-full text-sm',
  th: 'text-left text-[11px] font-medium uppercase tracking-wide text-zinc-400 pb-2',
  thr: 'text-right text-[11px] font-medium uppercase tracking-wide text-zinc-400 pb-2',
  td: 'py-2 border-t border-zinc-100 font-medium',
  tdr: 'py-2 border-t border-zinc-100 text-right tnum',
}

export function EvidenceTable({ evidence }: { evidence: EvidenceItem[] }) {
  if (evidence.length === 0) {
    return (
      <div className={s.wrap}>
        <div className={s.title}>근거 · 실측 인용 (LLM이 지어낸 값 아님)</div>
        <p className="text-sm text-zinc-500 leading-relaxed">
          단일 변수 임계(|z| ≥ 2σ)를 넘는 변수 없음 — 게이트는 <span className="font-medium text-ink">다변량 조합</span>으로 발화.
          한 변수로 환원되는 근거가 없어 인용을 지어내지 않음.
        </p>
      </div>
    )
  }
  return (
    <div className={s.wrap}>
      <div className={s.title}>근거 · 실측 인용 (LLM이 지어낸 값 아님)</div>
      <table className={s.table}>
        <thead>
          <tr>
            <th className={s.th}>변수</th>
            <th className={s.thr}>관측</th>
            <th className={s.thr}>정상평균</th>
            <th className={s.thr}>정상 SD</th>
            <th className={s.thr}>z(σ)</th>
          </tr>
        </thead>
        <tbody>
          {evidence.map((e) => (
            <tr key={e.var}>
              <td className={s.td}>{e.var}</td>
              <td className={s.tdr}>{e.observed}</td>
              <td className={s.tdr}>{e.normalMean}</td>
              <td className={s.tdr}>{e.normalSd}</td>
              <td className={s.tdr}>{e.z > 0 ? '+' : ''}{e.z}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
