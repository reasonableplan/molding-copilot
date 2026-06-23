"""12_ecod_compare.py — ECOD(PyOD) vs Mahalanobis, 같은 conformal 게이트로 공정 비교.
둘 다 정상으로 fit → anomaly score → split-conformal(α=0.05) → 진짜결함 recall / 모드별 / 오탐율.
실행: python eda/12_ecod_compare.py   산출: outputs/12_ecod_compare.txt
"""
import os, sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from pyod.models.ecod import ECOD
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.conformal import ConformalGate

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "12_ecod_compare.txt"); L = []; log = lambda s="": L.append(str(s))

df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce'); live = [c for c in feats if X[c].nunique() > 1]
X = X[live].fillna(X[live].median()).values; reason = cn7['Reason'].values; is_normal = (cn7['PassOrFail']=='Y').values
real = np.isin(reason, ['가스','미성형'])

nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
a, b = int(len(nidx)*0.5), int(len(nidx)*0.75)
fit_i, cal_i, te_i = nidx[:a], nidx[a:b], nidx[b:]

# --- Mahalanobis scorer ---
sc = StandardScaler().fit(X[fit_i]); cov = LedoitWolf().fit(sc.transform(X[fit_i]))
maha = lambda idx: cov.mahalanobis(sc.transform(X[idx]))
# --- ECOD scorer (PyOD, 파라미터 없음, 표준화 불필요) ---
ecod = ECOD(); ecod.fit(X[fit_i])
ec = lambda idx: ecod.decision_function(X[idx])

ALPHA = 0.05
def evaluate(name, scorer):
    g = ConformalGate(ALPHA).fit(scorer(cal_i))
    fpr = g.is_anomaly(scorer(te_i)).mean()
    rec = lambda m: g.is_anomaly(scorer(np.where(m)[0])).mean()
    log(f"\n[{name}]  test-정상 오탐율={fpr*100:.1f}% (α=5%)")
    log(f"   진짜결함(가스+미성형) recall = {rec(real)*100:.0f}%")
    for md in ['가스','미성형','초기허용불량']:
        m = reason==md
        log(f"     {md}: {rec(m)*100:.0f}% ({int(g.is_anomaly(scorer(np.where(m)[0])).sum())}/{int(m.sum())})")
    return g

log("=== ECOD(PyOD) vs Mahalanobis — 같은 conformal 게이트(α=0.05) ===")
log(f"정상 split fit={len(fit_i)} calib={len(cal_i)} test={len(te_i)} | 진짜결함={int(real.sum())}")
evaluate("Mahalanobis (covariance)", maha)
evaluate("ECOD (per-feature 꼬리확률)", ec)

# --- ECOD 해석가능성: 가스 1건의 변수별 기여 (꼬리 점수 큰 변수) ---
gas_i = np.where(reason=='가스')[0]
ecod.fit(X[fit_i])
Oall = ecod.O  # fit 데이터의 차원별 점수 — 새 샘플은 재계산 필요
# 새 샘플 차원별 점수: ECOD 내부와 동일하게 좌/우 꼬리 -log CDF 합의 차원분해
from scipy.stats import skew
Xf = X[fit_i]
def ecod_dims(x):
    # 좌우 꼬리확률(경험적 CDF) -> 차원별 outlier 점수
    n = len(Xf); s = np.zeros(len(live))
    for j in range(len(live)):
        col = Xf[:, j]
        Fl = (np.sum(col <= x[j]) + 1) / (n + 1)      # 좌측 꼬리
        Fr = (np.sum(col >= x[j]) + 1) / (n + 1)      # 우측 꼬리
        s[j] = -np.log(min(Fl, Fr))
    return s
g0 = X[gas_i[0]]; sd = ecod_dims(g0)
top = np.argsort(-sd)[:4]
log("\n[ECOD 해석] 가스 1건의 변수별 꼬리 점수 top4 (클수록 극단):")
for j in top:
    log(f"   {live[j]:<24} 점수 {sd[j]:.2f}  (관측 {g0[j]:.1f})")
log("→ ECOD도 금형온도를 극단으로 지목하면 grounding 근거로 직결 (Mahalanobis와 같은 원인 변수).")

open(OUT,"w",encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
