// 측정된 신뢰 탭 — CW 대조 막대(우리=라임/vanilla=다크) + 지표 카드.
import { useDiagnosisStore } from '../../shared/store/diagnosis.store'

const s = {
  head: 'rounded-card bg-white shadow-card p-6',
  title: 'text-lg font-bold',
  sub: 'text-sm text-zinc-500 mt-1 mb-6',
  barRow: 'flex items-end justify-center gap-12 h-44 border-b border-zinc-200',
  barCol: 'w-28 flex flex-col items-center justify-end h-full',
  barValue: 'text-lg font-extrabold tnum mb-2',
  labelRow: 'flex justify-center gap-12 mt-3',
  barLabel: 'w-28 text-center text-sm font-medium text-zinc-600',
  baseline: 'text-[11px] text-zinc-400 text-right mt-2',
  delta: 'w-20 self-center flex flex-col items-center',
  deltaNum: 'text-3xl font-extrabold tnum text-accentdk leading-none',
  deltaLabel: 'text-[11px] text-zinc-400 mt-1 text-center',
  metrics: 'grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4',
  metric: 'rounded-card bg-white shadow-card p-5',
  mLabel: 'text-xs font-medium uppercase tracking-wide text-zinc-400',
  mValue: 'text-3xl font-extrabold tnum mt-1',
  mNote: 'text-xs text-zinc-400 mt-2 leading-relaxed',
  pill: 'inline-block rounded-full bg-accent px-2.5 py-0.5 text-xs font-semibold text-ink',
}

export function TrustContainer() {
  const trust = useDiagnosisStore((st) => st.trust)
  if (!trust) return <p className="text-sm text-zinc-400 py-10 text-center">불러오는 중…</p>

  const max = Math.max(trust.cwOurs, trust.cwVanilla)
  const h = (v: number) => `${(v / max) * 100}%`
  const delta = Math.round(trust.cwVanilla - trust.cwOurs)

  return (
    <div className="flex flex-col gap-4">
      <div className={s.head}>
        <div className={s.title}>핵심 — 엔진 有(우리) vs 無(vanilla) · 같은 gemma3:4b</div>
        <div className={s.sub}>
          정상 40건을 불량이라 단정한 비율(confidently-wrong) ·{' '}
          <span className={s.pill}>차이 60%p · 95%CI[{trust.cwDiffCi[0]},{trust.cwDiffCi[1]}] · 0 미포함=유의</span>
        </div>
        <div className={s.barRow}>
          <div className={s.barCol}>
            <span className={s.barValue} style={{ color: '#a9d62a' }}>{trust.cwOurs}%</span>
            <div className="w-28 rounded-t-xl bg-accent" style={{ height: h(trust.cwOurs) }} />
          </div>
          <div className={s.delta}>
            <span className={s.deltaNum}>−{delta}%p</span>
            <span className={s.deltaLabel}>오진 감소</span>
          </div>
          <div className={s.barCol}>
            <span className={s.barValue}>{trust.cwVanilla}%</span>
            <div className="w-28 rounded-t-xl bg-ink" style={{ height: h(trust.cwVanilla) }} />
          </div>
        </div>
        <div className={s.labelRow}>
          <span className={s.barLabel}>우리 (엔진)</span>
          <span className="w-20" />
          <span className={s.barLabel}>vanilla</span>
        </div>
        <div className={s.baseline}>기준선 = 정상 40건 (낮을수록 좋음)</div>
      </div>

      <div className={s.metrics}>
        <div className={s.metric}>
          <div className={s.mLabel}>검출 recall</div>
          <div className={s.mValue}>{trust.recallSup}%</div>
          <div className={s.mNote}>비지도 {trust.recallUnsup}% → 지도 게이트 · 95%CI[{trust.recallCi[0]},{trust.recallCi[1]}]</div>
        </div>
        <div className={s.metric}>
          <div className={s.mLabel}>정상 오탐율 (보장)</div>
          <div className={s.mValue}>{trust.fprGuarantee}%</div>
          <div className={s.mNote}>≤ α={trust.alpha}% 보장 · conformal</div>
        </div>
        <div className={s.metric}>
          <div className={s.mLabel}>RG3 일반화</div>
          <div className={s.mValue}>{trust.rg3Fpr}%</div>
          <div className={s.mNote}>보장은 전이 / recall {trust.rg3Recall}%는 라인 의존</div>
        </div>
      </div>
    </div>
  )
}
