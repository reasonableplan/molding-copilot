"""06_conformal_gate.py — 3a: conformal '모른다' 게이트 검증.
Mahalanobis 거리(정상으로부터) = anomaly score → conformal p-value → 보장된 오탐율 게이트.
검증: test-정상 실제 오탐율 <= alpha 인가 / 진짜결함(가스+미성형) recall / step2 PCA-AE 대비.
실행: python eda/06_conformal_gate.py   산출: outputs/06_conformal.txt
"""
import os, sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.conformal import ConformalGate

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "06_conformal.txt")
L = []; log = lambda s="": L.append(str(s))

df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce')
live = [c for c in feats if X[c].nunique() > 1]; X = X[live].fillna(X[live].median())
is_normal = (cn7['PassOrFail']=='Y').values; reason = cn7['Reason'].values
real = np.isin(reason, ['가스','미성형'])

# 정상 3분할: fit(50%) / calib(25%) / test(25%)
nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
a, b = int(len(nidx)*0.5), int(len(nidx)*0.75)
fit_i, cal_i, te_i = nidx[:a], nidx[a:b], nidx[b:]

# Mahalanobis 점수 (정상 fit)
sc = StandardScaler().fit(X.iloc[fit_i])
cov = LedoitWolf().fit(sc.transform(X.iloc[fit_i]))
score = lambda idx: cov.mahalanobis(sc.transform(X.iloc[idx]))   # 제곱 마할라노비스

log(f"정상 split: fit={len(fit_i)} calib={len(cal_i)} test={len(te_i)} | 진짜결함={real.sum()}")
log("Mahalanobis + split-conformal 게이트 — 보장은 marginal(여러 split 평균)이므로 K=300 split로 검증:\n")
log(f"{'alpha':>6} | {'평균 test-정상 오탐율':>17} | {'95% 분위':>8} | {'보장(평균<=α)':>11} | {'평균 진짜recall':>13}")
real_idx = np.where(real)[0]
for alpha in (0.05, 0.01):
    fprs, recs = [], []
    for k in range(300):
        rs = np.random.default_rng(1000+k)
        ni = nidx.copy(); rs.shuffle(ni)
        fi, ci, ti = ni[:a], ni[a:b], ni[b:]
        scl = StandardScaler().fit(X.iloc[fi]); cv = LedoitWolf().fit(scl.transform(X.iloc[fi]))
        sco = lambda idx: cv.mahalanobis(scl.transform(X.iloc[idx]))
        g = ConformalGate(alpha).fit(sco(ci))
        fprs.append(g.is_anomaly(sco(ti)).mean()); recs.append(g.is_anomaly(sco(real_idx)).mean())
    fprs, recs = np.array(fprs), np.array(recs)
    ok = "OK" if fprs.mean() <= alpha + 1e-9 else "위반!"
    log(f"{alpha:>6.2f} | {fprs.mean()*100:>16.2f}% | {np.percentile(fprs,95)*100:>6.2f}% | {ok:>11} | {recs.mean()*100:>12.0f}%")

# 모드별 recall @ alpha=0.05 (단일 대표 split)
gate = ConformalGate(0.05).fit(score(cal_i))
log("\n@alpha=0.05 모드별 recall:")
for m in ['가스','미성형','초기허용불량']:
    idx = np.where(reason==m)[0]
    log(f"   {m}: {gate.is_anomaly(score(idx)).mean()*100:.0f}%  ({int(gate.is_anomaly(score(idx)).sum())}/{len(idx)})")

log("\n대조 — step2 PCA-AE 진짜결함 recall@FPR5% = 42% (오탐율 통제 안 됨).")
log("conformal 이점: 오탐율이 '관측'이 아니라 'α로 보장'. 임계를 데이터로 안 고르고 보장에서 역산.")
open(OUT,"w",encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
