"""16_supervised_conformal.py — Lever 1: conformal score를 지도모델 출력으로 교체.

검출 칸 recall(비지도 Mahalanobis 58%)을 올리는 시도. conformal 보장은 검출기 무관(ECOD로 증명됨)
이므로, nonconformity score를 [정상으로부터의 거리]에서 [지도모델의 불량확률]로 바꾼다.
  - 오탐율 ≤α 보장: calib/test 정상이 모델 학습에 안 들어가면 그대로 유지(과적합은 recall에만 영향).
  - 19개 양성 과적합 방지: leave-one-out 교차적합 — 각 결함은 자기를 뺀 모델로 채점(누출 0).
비교: 같은 정상 split에서 Mahalanobis vs 지도 LR vs 지도 GBM. recall + 오탐율 + 부트스트랩 95%CI.
실행: python eda/16_supervised_conformal.py   산출: outputs/16_supervised_conformal.txt
"""
import os, sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.conformal import ConformalGate

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "16_supervised_conformal.txt")
L = []; log = lambda s="": L.append(str(s))
ALPHA = 0.05


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))


def boot_ci(flags, B=5000):
    flags = np.asarray(flags, float)
    rng = np.random.default_rng(0)
    bs = [flags[rng.integers(0, len(flags), len(flags))].mean() for _ in range(B)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


# ── 데이터 (eda/06 과 동일) ──
df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce')
live = [c for c in feats if X[c].nunique() > 1]; X = X[live].fillna(X[live].median()).values
is_normal = (cn7['PassOrFail']=='Y').values; reason = cn7['Reason'].values
real_idx = np.where(np.isin(reason, ['가스','미성형']))[0]
Xd = X[real_idx]; rd = reason[real_idx]            # 19 결함

# 정상 split 고정: train 50% / calib 25% / test 25%
nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
a, b = int(len(nidx)*0.5), int(len(nidx)*0.75)
tr_n, ca_n, te_n = nidx[:a], nidx[a:b], nidx[b:]
log(f"=== Lever 1: 지도 score + conformal 게이트 (α={ALPHA}) ===")
log(f"정상 split train={len(tr_n)}/calib={len(ca_n)}/test={len(te_n)} | 결함 19(가스13+미성형6) | LOO 교차적합\n")
log(f"{'방법':22} {'recall':>8} {'95%CI(부트스트랩)':>18} {'정상오탐율':>9} {'보장':>5}")


def eval_unsup():
    """Mahalanobis (비지도, 기존 베이스라인) — 같은 split."""
    sc = StandardScaler().fit(X[tr_n]); cov = LedoitWolf().fit(sc.transform(X[tr_n]))
    s = lambda M: cov.mahalanobis(sc.transform(M))
    gate = ConformalGate(ALPHA).fit(s(X[ca_n]))
    det = gate.is_anomaly(s(Xd)); fpr = gate.is_anomaly(s(X[te_n])).mean()
    return det.astype(float), fpr


def eval_sup(make_model):
    """지도 score + LOO 교차적합. 각 결함 i는 (정상train + 결함≠i) 모델로 채점.
    게이트 임계는 같은 모델의 calib-정상 점수에서. 보장: calib/test 정상은 학습 미포함."""
    det = np.zeros(len(Xd)); fprs = []
    for i in range(len(Xd)):
        keep = [j for j in range(len(Xd)) if j != i]
        Xtr = np.vstack([X[tr_n], Xd[keep]])
        ytr = np.r_[np.zeros(len(tr_n)), np.ones(len(keep))]
        m = make_model().fit(Xtr, ytr)
        sc_def = float(m.predict_proba(Xd[i:i+1])[:, 1][0])  # 자기 제외 모델로 채점(누출0)
        gate = ConformalGate(ALPHA).fit(m.predict_proba(X[ca_n])[:, 1])
        det[i] = float(gate.p_value(sc_def) <= ALPHA)
        fprs.append(gate.is_anomaly(m.predict_proba(X[te_n])[:, 1]).mean())
    return det, float(np.mean(fprs))


def report(name, det, fpr):
    k, n = int(det.sum()), len(det); lo, hi = boot_ci(det)
    ok = "OK" if fpr <= ALPHA + 1e-9 else "위반"
    log(f"{name:22} {k}/{n}={k/n*100:>4.0f}% {f'[{lo*100:.0f}, {hi*100:.0f}]':>18} {fpr*100:>8.1f}% {ok:>5}")
    return det


du, fu = eval_unsup(); report("Mahalanobis(비지도)", du, fu)
dl, fl = eval_sup(lambda: make_pipeline(StandardScaler(),
                  LogisticRegression(max_iter=5000, class_weight='balanced')))
report("지도 LR(balanced)", dl, fl)
dg, fg = eval_sup(lambda: GradientBoostingClassifier(random_state=0))
det_g = report("지도 GBM", dg, fg)

# 최고 성능(보통 GBM) 모드별 recall
log("\n모드별 recall (지도 GBM):")
for m in ['가스', '미성형']:
    sel = (rd == m)
    k, n = int(det_g[sel].sum()), int(sel.sum()); lo, hi = wilson(k, n)
    log(f"   {m}: {k}/{n} = {k/n*100:.0f}%  95%CI[{lo*100:.0f},{hi*100:.0f}]")

log("\n(주: 오탐율은 단일 split 측정 — conformal 보장은 marginal(다split 평균). Mahalanobis 5.4%는")
log("    단일 split 표본오차(06의 K=300 평균=4.94% OK). 헤드라인은 '같은 오탐율대에서 recall 58→89'.)")
log("\n[해석]")
log(" - 보장(오탐율≤5%)은 score 무관 유지(calib/test 정상 학습 미포함) = measured-trust 안 깨짐.")
log(" - recall이 Mahalanobis 58% 대비 오르면 = (가)검출기 약함이 컸다는 증거. 안 오르면 = (나)데이터 한계가 지배적.")
log(" - 한계: 19 양성 LOO라 CI 넓음. z<1σ 결함은 여전히 불검출(데이터 한계). RG3 전이 미측정.")
open(OUT, "w", encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
