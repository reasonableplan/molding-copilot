"""14_bootstrap_ci.py — 핵심 지표에 신뢰구간(불확실성 정직 보고).

이 프로젝트 브랜드 = "양성 19개뿐 → 분산을 정직하게". 그런데 헤드라인 숫자(recall 58%,
CW 8% vs 68%, 모드적중)에 CI가 없었음. 양성이 적을수록 점추정은 위험 → CI로 정직하게.

방법: (1) 비율 지표는 Wilson score 95% CI(소표본에 적합). (2) conformal recall은 19개 결함을
부트스트랩 재표집해 분포까지. (3) 정상 오탐율은 큰 n이라 tight할 것 — 대조로 보임.
실행: python eda/14_bootstrap_ci.py   산출: outputs/14_bootstrap_ci.txt
"""
import os, sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.conformal import ConformalGate

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "14_bootstrap_ci.txt")
L = []; log = lambda s="": L.append(str(s))


def wilson(k, n, z=1.96):
    """Wilson score 95% CI (정규근사보다 소표본에 정확, k=0/n에서도 안전)."""
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))


def line(name, k, n):
    lo, hi = wilson(k, n)
    log(f"   {name:32} {k:>3}/{n:<4} = {k/n*100:>5.1f}%   95%CI [{lo*100:>5.1f}, {hi*100:>5.1f}]")


# ── 데이터 + Mahalanobis/conformal (eda/06 과 동일 셋업) ──
df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce')
live = [c for c in feats if X[c].nunique() > 1]; X = X[live].fillna(X[live].median())
is_normal = (cn7['PassOrFail']=='Y').values; reason = cn7['Reason'].values
real = np.isin(reason, ['가스','미성형'])

nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
a, b = int(len(nidx)*0.5), int(len(nidx)*0.75)
fit_i, cal_i, te_i = nidx[:a], nidx[a:b], nidx[b:]
sc = StandardScaler().fit(X.iloc[fit_i]); cov = LedoitWolf().fit(sc.transform(X.iloc[fit_i]))
score = lambda idx: cov.mahalanobis(sc.transform(X.iloc[idx]))
gate = ConformalGate(0.05).fit(score(cal_i))

log("=== 핵심 지표 95% 신뢰구간 (양성 19개 = 모든 CI 넓음 = 정직한 한계) ===\n")

# ── 1) 검출 칸: conformal recall + 모드별 (Wilson) + 부트스트랩 ──
real_idx = np.where(real)[0]
det = gate.is_anomaly(score(real_idx))          # 19개 결함 각각 탐지 여부 (bool)
gas_idx = np.where(reason=='가스')[0]; short_idx = np.where(reason=='미성형')[0]
log("[검출] conformal 게이트 recall @α=0.05 (대표 split):")
line("진짜결함(가스+미성형)", int(det.sum()), len(real_idx))
line("가스", int(gate.is_anomaly(score(gas_idx)).sum()), len(gas_idx))
line("미성형", int(gate.is_anomaly(score(short_idx)).sum()), len(short_idx))
# 부트스트랩: 19개 결함을 재표집 → recall 분포
boot = []
for k in range(5000):
    rs = np.random.default_rng(k)
    boot.append(det[rs.integers(0, len(det), len(det))].mean())
boot = np.array(boot)
log(f"   부트스트랩(5000회, 결함 재표집) recall: 중앙 {np.median(boot)*100:.1f}% / "
    f"2.5~97.5%분위 [{np.percentile(boot,2.5)*100:.1f}, {np.percentile(boot,97.5)*100:.1f}]")
log("")

# ── 2) 정상 오탐율 (큰 n → tight, 대조) ──
fp = gate.is_anomaly(score(te_i))
log("[검출] 정상 test 오탐율 (대조 — n 크면 CI 좁음):")
line("오탐율 @α=0.05", int(fp.sum()), len(te_i))
log("")

# ── 3) 설명/정직성: 저장된 카운트에 CI (LLM 재실행 없이 비율 CI) ──
log("[정직성] 09_faithfulness 저장 결과에 CI (비율이라 LLM 재실행 불필요):")
log("  모드 적중 (발화건 기준, 07_detector):")
line("우리(엔진) 모드적중", 9, 11)        # 07: 발화11 적중9 (모드별 분해는 미저장 → 전체만)
log("  정상 40건 confidently-wrong (기권이 정답):")
line("우리(엔진) CW", 3, 40)             # 09: 8% = 3/40
line("vanilla CW", 27, 40)              # 09: 68% = 27/40
# 두 비율 차이 CI (독립 비율차 정규근사)
p1,n1,p2,n2 = 3/40,40,27/40,40
se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2); diff=p2-p1
log(f"   → CW 차이(vanilla-우리) = {diff*100:.0f}%p  95%CI [{(diff-1.96*se)*100:.0f}, {(diff+1.96*se)*100:.0f}]p (0 미포함 → 유의)")
log("")

log("[해석] 진짜결함 recall 58%의 CI가 [~36, ~77]로 넓음 = 양성 19개의 정직한 현실(점추정 과신 금지).")
log("       반면 정상 오탐율·CW 차이는 n이 커 CI가 좁아 결론 견고. = 어디는 확실/어디는 불확실을 분리 보고.")
open(OUT,"w",encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
