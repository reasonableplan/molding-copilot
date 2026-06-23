"""17_hybrid_ensemble.py — Lever 1 강화: 하이브리드(지도+비지도) 검출, 안전망 검증.

지도 게이트(eda/16, recall 89%)의 약점 = 라벨된 결함모드만 앎(처음 보는 결함=unknown unknown에 약함).
보강 방식 탐색:
  (1) 두 score를 표준화해 합/최대로 결합 후 단일 conformal 게이트 → 실패(결합 null 분포가 부풀어 임계↑,
      지도가 잡던 것까지 놓침. sum=79%, max=68% < 지도 89%). = 점수 블렌딩은 conformal에서 역효과.
  (2) ★두 게이트의 합집합(α 예산 분할, Bonferroni): 지도 게이트(α1) OR Mahalanobis 게이트(α2), α1+α2=α.
      각 게이트가 자기 null로 임계를 잡아 지도 recall 보존 + Mahalanobis가 novel 안전망. 합집합 오탐율 ≤ α1+α2.

검증:
  A) 전체 19결함(LOO): 지도-only(α) vs 합집합(α1+α2) — 같은 오탐율 예산에서 known recall 비교.
  B) ★novel-모드 leave-out: 한 모드 통째로 빼고 학습 → 지도-only는 무너지고 합집합이 회복하나(안전망).
실행: python eda/17_hybrid_ensemble.py   산출: outputs/17_hybrid_ensemble.txt
"""
import os, sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.conformal import ConformalGate

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "17_hybrid_ensemble.txt")
L = []; log = lambda s="": L.append(str(s))
ALPHA = 0.05
A1, A2 = 0.04, 0.01          # 예산 분할: 지도 게이트 0.04 + Mahalanobis 게이트 0.01 = 0.05


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))


# ── 데이터 ──
df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce')
live = [c for c in feats if X[c].nunique() > 1]; X = X[live].fillna(X[live].median()).values
is_normal = (cn7['PassOrFail']=='Y').values; reason = cn7['Reason'].values
real_mask = np.isin(reason, ['가스','미성형'])
Xd = X[real_mask]; rd = reason[real_mask]

nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
a, b = int(len(nidx)*0.5), int(len(nidx)*0.75)
tr_n, ca_n, te_n = nidx[:a], nidx[a:b], nidx[b:]

# 비지도 Mahalanobis (정상 train fit — 고정) + 그 게이트(α2)
msc = StandardScaler().fit(X[tr_n]); mcov = LedoitWolf().fit(msc.transform(X[tr_n]))
maha = lambda M: mcov.mahalanobis(msc.transform(M))
m_ca, m_te, m_def = maha(X[ca_n]), maha(X[te_n]), maha(Xd)
gate_m = ConformalGate(A2).fit(m_ca)
def hit_m(scores): return np.array([gate_m.p_value(float(s)) <= A2 for s in np.atleast_1d(scores)])


def fit_sup(defect_X):
    m = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=0))
    Xtr = np.vstack([X[tr_n], defect_X]); ytr = np.r_[np.zeros(len(tr_n)), np.ones(len(defect_X))]
    return m.fit(Xtr, ytr)


# ════════ A) 전체 19결함 (LOO) — 지도-only(α) vs 합집합(α1+α2) ════════
log("=== A) 전체 19결함 (LOO) — 지도-only(α=0.05) vs 하이브리드 합집합(지도 0.04 OR Maha 0.01) ===")
det_s = np.zeros(len(Xd)); det_h = np.zeros(len(Xd)); fpr_s = []; fpr_h = []
for i in range(len(Xd)):
    keep = [j for j in range(len(Xd)) if j != i]
    m = fit_sup(Xd[keep])
    s_ca, s_te = m.predict_proba(X[ca_n])[:,1], m.predict_proba(X[te_n])[:,1]
    s_di = m.predict_proba(Xd[i:i+1])[:,1]
    g05 = ConformalGate(ALPHA).fit(s_ca); g_s = ConformalGate(A1).fit(s_ca)
    det_s[i] = g05.p_value(float(s_di[0])) <= ALPHA                        # 지도-only α
    fpr_s.append(g05.is_anomaly(s_te).mean())
    # 합집합: 지도(α1) OR Maha(α2)
    det_h[i] = (g_s.p_value(float(s_di[0])) <= A1) or bool(hit_m(m_def[i])[0])
    union_te = (np.array([g_s.p_value(float(s)) <= A1 for s in s_te])) | hit_m(m_te)
    fpr_h.append(union_te.mean())

def row(name, det, fpr):
    k, n = int(det.sum()), len(det); lo, hi = wilson(k, n)
    log(f"   {name:28} recall {k}/{n}={k/n*100:>3.0f}%  95%CI[{lo*100:.0f},{hi*100:.0f}]  오탐율 {np.mean(fpr)*100:.1f}%")
row("지도-only (α=0.05)", det_s, fpr_s)
row("하이브리드 합집합 (0.04+0.01)", det_h, fpr_h)
log(f"   (참고: 비지도 Mahalanobis 단독 α=0.05 = 58%. 점수블렌딩(sum/max)은 79/68%로 역효과 → 합집합 채택.)")
log("   → 합집합이 지도 recall을 거의 유지하면 = 안전망을 오탐율 예산 안에서 추가(거의 공짜).\n")

# ════════ B) novel-모드 leave-out — 안전망 핵심 검증 ════════
log("=== B) ★novel-모드 leave-out (한 모드 통째로 빼고 학습 → 처음 보는 결함) ===")
for novel in ['가스', '미성형']:
    seen = '미성형' if novel == '가스' else '가스'
    Xseen, Xnovel = Xd[rd == seen], Xd[rd == novel]
    m = fit_sup(Xseen)
    s_ca, s_nv = m.predict_proba(X[ca_n])[:,1], m.predict_proba(Xnovel)[:,1]
    g_s = ConformalGate(A1).fit(s_ca)
    mn = maha(Xnovel)
    ds = np.array([g_s.p_value(float(s)) <= A1 for s in s_nv])             # 지도-only(α1)
    dm = hit_m(mn)                                                        # Maha(α2=0.01)
    dh = ds | dm                                                          # 합집합(0.04+0.01)
    gm05 = ConformalGate(0.05).fit(m_ca)
    dm5 = np.array([gm05.p_value(float(s)) <= 0.05 for s in mn])          # Maha 풀예산(0.05)
    n = len(Xnovel)
    log(f"   [novel='{novel}' (학습엔 '{seen}'만)]  novel {n}건:")
    log(f"      지도-only(0.04)        {int(ds.sum())}/{n} = {ds.mean()*100:>3.0f}%   ← 처음 본 결함이라 약함")
    log(f"      Maha 안전망(0.01)      {int(dm.sum())}/{n} = {dm.mean()*100:>3.0f}%   ← 예산 작음→약함")
    log(f"      합집합(0.04+0.01)      {int(dh.sum())}/{n} = {dh.mean()*100:>3.0f}%")
    log(f"      [다이얼업] Maha 풀예산(0.05) {int(dm5.sum())}/{n} = {dm5.mean()*100:>3.0f}%   ← FPR 더 쓰면 novel↑")
    log("")

log("[해석]")
log(" - 점수 블렌딩(sum/max)은 conformal에서 역효과(결합 null 부풀어 임계↑) → 합집합(게이트 OR) 채택.")
log(" - A: 합집합이 같은 오탐율 예산(0.05)에서 지도 recall 거의 유지 = 안전망 추가 비용 작음.")
log(" - B: 지도-only는 novel서 무너짐(가스 15%). Maha 안전망은 unknown을 잡을 능력 有(풀예산 54%)지만")
log("   작은 예산(0.01)선 약함(15%=회복못함). = 안전망 세기는 준 FPR 예산에 비례.")
log(" - ★핵심 trade-off(정직): 고정 FPR 예산서 known recall(지도)과 novel coverage(Maha)는 경합.")
log("   89% known 유지하려 지도 0.04→안전망 0.01=약함. novel 강하게(54%)면 FPR 더 써야 = 안전망은 다이얼(공짜 아님).")
log(" - z<1σ 결함은 어느 예산에서도 불검출(데이터 한계, 모델로 못 풂).")
open(OUT, "w", encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
