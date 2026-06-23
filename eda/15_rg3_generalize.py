"""15_rg3_generalize.py — 두 번째 실데이터(RG3)로 일반화 검증.

RG3 = CN7과 다른 부품/금형의 실제 라인. labeled 1,256 shot, 진짜결함 32(가스22+미성형10),
cold-start 0. 같은 24 공정변수·같은 결함모드. 두 가지 정직한 검증:
  ① 방법 일반화: RG3 정상으로 새 calibration → conformal recall/오탐율이 RG3에서도 서나?
  ② 시그니처 순환성 깨기: 가스=과열·미성형=충전부족 시그니처는 CN7에서 유도됨(약한 순환성).
     CN7 시그니처를 RG3에 그대로 적용해 모드적중 → 맞으면 외부검증(순환 반증), 틀리면 라인별 재학습 필요.
실행: python eda/15_rg3_generalize.py   산출: outputs/15_rg3_generalize.txt
"""
import os, sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.conformal import ConformalGate
from src.grounding import _VAR2MODE          # CN7에서 유도된 시그니처 (변경 없이 재사용)

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "15_rg3_generalize.txt")
L = []; log = lambda s="": L.append(str(s))


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (max(0, c-h), min(1, c+h))


# ── RG3 로드 (labeled_data 에서 RG3 부품군) ──
df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
rg3 = df[df['PART_NAME'].astype(str).str.startswith('RG3')].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in rg3.columns if c not in meta]
X = rg3[feats].apply(pd.to_numeric, errors='coerce')
live = [c for c in feats if X[c].nunique() > 1]; X = X[live].fillna(X[live].median())
is_normal = (rg3['PassOrFail']=='Y').values; reason = rg3['Reason'].values
real = np.isin(reason, ['가스','미성형'])

log("=== RG3 일반화 검증 (CN7과 다른 부품/금형의 실데이터) ===")
log(f"RG3: 총 {len(rg3)} shot | 정상 {is_normal.sum()} | 진짜결함 {real.sum()} (가스 {(reason=='가스').sum()}, 미성형 {(reason=='미성형').sum()}) | cold-start 0")
log(f"live 변수 {len(live)}개 (CN7과 동일 집합? {set(live)== set([c for c in feats])} — 상수열 제거 후)\n")

# ── ① 방법 일반화: RG3 정상으로 새 calibration ──
nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
a, b = int(len(nidx)*0.5), int(len(nidx)*0.75)
fit_i, cal_i, te_i = nidx[:a], nidx[a:b], nidx[b:]
sc = StandardScaler().fit(X.iloc[fit_i]); cov = LedoitWolf().fit(sc.transform(X.iloc[fit_i]))
score = lambda idx: cov.mahalanobis(sc.transform(X.iloc[idx]))
gate = ConformalGate(0.05).fit(score(cal_i))

real_idx = np.where(real)[0]
det = gate.is_anomaly(score(real_idx))
fp = gate.is_anomaly(score(te_i))
gas_idx = np.where(reason=='가스')[0]; sh_idx = np.where(reason=='미성형')[0]
rk, rn = int(det.sum()), len(real_idx); rlo, rhi = wilson(rk, rn)
flo, fhi = wilson(int(fp.sum()), len(te_i))
log(f"① 방법 일반화 (RG3 정상으로 fit={len(fit_i)}/calib={len(cal_i)}/test={len(te_i)}, α=0.05):")
log(f"   정상 test 오탐율 = {fp.mean()*100:.1f}%  95%CI[{flo*100:.1f},{fhi*100:.1f}]  (목표 ≤5%)")
log(f"   진짜결함 recall  = {rk}/{rn} = {rk/rn*100:.1f}%  95%CI[{rlo*100:.1f},{rhi*100:.1f}]")
log(f"      가스 {int(gate.is_anomaly(score(gas_idx)).sum())}/{len(gas_idx)} · 미성형 {int(gate.is_anomaly(score(sh_idx)).sum())}/{len(sh_idx)}")
log(f"   [대조 CN7] 오탐율 5.4% / recall 57.9%[36,77]  → RG3가 같은 궤도면 '방법 일반화' 성립\n")

# ── ② 시그니처 순환성 깨기: CN7 유도 시그니처를 RG3 발화건에 적용 ──
# detector 와 동일 로직: conformal 게이트 통과(발화) 건에 대해 |z|≥2 인용변수 → _VAR2MODE |z|가중 투표
mu = X.iloc[fit_i].mean(); sd = X.iloc[fit_i].std().replace(0, np.nan)
def predict_mode(i):
    z = ((X.iloc[i] - mu)/sd).dropna()
    cited = z.reindex(z.abs().sort_values(ascending=False).index)
    cited = cited[cited.abs() >= 2.0].head(3)
    w = {}
    for v in cited.index:
        m = _VAR2MODE.get(v)
        if m: w[m] = w.get(m, 0.0) + abs(z[v])
    return max(w, key=w.get) if w else None

spoke = real_idx[det]                      # 게이트 통과(발화)한 진짜결함
hit = tot = 0; per = {'가스': [0,0], '미성형': [0,0]}
for i in spoke:
    pred = predict_mode(i); actual = reason[i]
    if pred is not None:
        tot += 1; per[actual][1] += 1
        if pred == actual: hit += 1; per[actual][0] += 1
mlo, mhi = wilson(hit, tot) if tot else (0,0)
log(f"② 시그니처 순환성 깨기 (CN7 유도 시그니처 _VAR2MODE를 RG3에 그대로 적용):")
log(f"   발화 {len(spoke)}건 중 모드추정 발화 {tot}건, 적중 {hit}/{tot} = {hit/tot*100 if tot else 0:.0f}%  95%CI[{mlo*100:.0f},{mhi*100:.0f}]")
log(f"      가스 {per['가스'][0]}/{per['가스'][1]} · 미성형 {per['미성형'][0]}/{per['미성형'][1]}")
log(f"   [대조 CN7] 모드적중 81.8%[52,95]")
log("   → CN7 시그니처가 RG3에서도 적중하면: 시그니처가 부품 독립 = '약한 순환성' 한계 반증(외부검증).")
log("     안 맞으면: 시그니처가 CN7 특수 = 라인별 재유도 필요(정직한 일반화 한계).\n")

# RG3 가스/미성형 실제 분리도 — 시그니처 가정(가스=금형온도↑, 미성형=사출속도↑)이 RG3서도 성립?
log("[참고] RG3 자체 분리도 (시그니처 가정 검증 — CN7과 같은 물리인가):")
for m in ['가스', '미성형']:
    idx = np.where(reason==m)[0]
    z = ((X.iloc[idx].mean() - mu)/sd).dropna().sort_values(key=abs, ascending=False)
    top = ", ".join(f"{v}={z[v]:+.1f}σ" for v in z.index[:3])
    log(f"   [{m}] top3 z-편차: {top}")
open(OUT, "w", encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
