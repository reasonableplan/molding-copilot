// 진단 탭 — 카드 그리드 레이아웃.
import { useDiagnosisStore } from '../../shared/store/diagnosis.store'
import { Gauge } from '../../shared/components/Gauge'
import { ZBars } from '../../shared/components/ZBars'
import { VerdictCard } from './components/VerdictCard'
import { EvidenceTable } from './components/EvidenceTable'

const s = {
  grid: 'grid grid-cols-1 lg:grid-cols-2 gap-4',
  presc: 'rounded-card bg-accent/15 border border-accent/40 p-6',
  prescTitle: 'text-xs font-medium uppercase tracking-wide text-accentdk mb-2',
  prescBody: 'text-sm text-ink/80 leading-relaxed',
  hint: 'rounded-card bg-zinc-50 border border-zinc-200 p-6 text-sm text-zinc-400',
  loading: 'text-sm text-zinc-400 py-10 text-center',
}

export function DiagnosisContainer() {
  const diagnosis = useDiagnosisStore((st) => st.diagnosis)
  if (!diagnosis) return <p className={s.loading}>불러오는 중…</p>

  return (
    <div className="flex flex-col gap-4">
      <VerdictCard diagnosis={diagnosis} />
      <div className={s.grid}>
        <Gauge pValue={diagnosis.pValue} alpha={diagnosis.alpha} anomaly={diagnosis.isAnomaly} />
        <ZBars bars={diagnosis.zbars} />
      </div>
      {diagnosis.isAnomaly ? (
        <>
          <EvidenceTable evidence={diagnosis.evidence} />
          {diagnosis.prescription && (
            <div className={s.presc}>
              <div className={s.prescTitle}>처방</div>
              <div className={s.prescBody}>{diagnosis.prescription}</div>
            </div>
          )}
        </>
      ) : (
        <div className={s.hint}>게이트가 닫혀 LLM을 거치지 않습니다 — 결정론적 기권.</div>
      )}
    </div>
  )
}
