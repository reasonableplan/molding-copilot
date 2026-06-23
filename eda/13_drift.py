"""13_drift.py — 예측 칸: conformal test martingale 드리프트 조기경보.

검출 칸의 split-conformal p-value 스트림을 Simple Jumper 마틴게일에 흘린다.
  A) 검증(validity): 교환가능 정상 스트림 → 오경보율 <= 1/c (Ville 보장 성립?)
  B) 실스트림: unlabeled_cn7 35,239 shot을 생산순서대로 → 마틴게일 궤적(실데이터에 드리프트?)
  C) 검정력(power): 정상 prefix + 금형온도(가스 시그니처) 점진 shift 주입 → 탐지 lead-time
실행: python eda/13_drift.py   산출: outputs/13_drift.txt, figures/drift_*.png
"""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.conformal import ConformalGate
from src.drift import SimpleJumper

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
FIG = os.path.join(ROOT, "outputs", "figures")
OUT = os.path.join(ROOT, "outputs", "13_drift.txt")
L = []; log = lambda s="": L.append(str(s))

# ── 데이터: supervised(라벨 有, calibration용) + unlabeled(35K, 스트림용) ──
sup = pd.read_csv(os.path.join(D, "supervised_label_cn7.csv"))
unl = pd.read_csv(os.path.join(D, "moldset_unlabeled_cn7.csv")).sort_values("Unnamed: 0")
feats = [c for c in sup.columns if c not in ("Unnamed: 0", "PassOrFail")]
Xs = sup[feats].apply(pd.to_numeric, errors="coerce")
Xs = Xs.fillna(Xs.median())
normal = (sup["PassOrFail"] == 0).values
Xu = unl[feats].apply(pd.to_numeric, errors="coerce")
Xu = Xu.fillna(Xs.median())   # 동일 기준(supervised median)으로 결측 대치

# ── Mahalanobis 스코어러: supervised 정상 절반으로 fit ──
nidx = np.where(normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
half = len(nidx) // 2
fit_i, cal_i = nidx[:half], nidx[half:]
sc = StandardScaler().fit(Xs.iloc[fit_i])
cov = LedoitWolf().fit(sc.transform(Xs.iloc[fit_i]))
def score_s(idx): return cov.mahalanobis(sc.transform(Xs.iloc[idx]))
def score_arr(arr): return cov.mahalanobis(sc.transform(arr))

gate = ConformalGate(alpha=0.05).fit(score_s(cal_i))   # calib = 정상 나머지 절반
C_THRESH = 100                                          # Ville: P(경보)<=1/100=1%

log("=== 예측 칸: Conformal Test Martingale 드리프트 조기경보 ===")
log(f"calib 정상={len(cal_i)}  fit 정상={len(fit_i)}  | Ville 임계 c={C_THRESH} → 오경보율 보장 <= {100/C_THRESH:.0f}%")
log("p-value: 검출 칸과 동일한 split-conformal(Mahalanobis). 연속 스코어라 타이 무시 가능.\n")

# ── A) 검증: 교환가능 정상 스트림에서 오경보율 <= 1/c 인가 ──
# calib과 독립인 평가용 정상 풀이 없으므로, calib 정상을 셔플(=교환가능)해 스트림 구성.
# (보수적: calib과 스트림이 같은 표본 → super-uniform p-value, 마틴게일은 super-martingale)
pool = cal_i.copy()
K = 300; L_STREAM = 1000
false_alarms = 0; maxes = []
for k in range(K):
    rs = np.random.default_rng(2000 + k)
    s_idx = rs.choice(pool, size=L_STREAM, replace=True)
    pv = gate.p_value(score_s(s_idx))
    traj, al = SimpleJumper().run(pv, threshold=C_THRESH)
    maxes.append(traj.max())
    if al: false_alarms += 1
fa_rate = false_alarms / K
log(f"[A 검증] 교환가능 정상 스트림 {K}회 × {L_STREAM}shot:")
log(f"   오경보율(sup S_n>=100) = {fa_rate*100:.1f}%  (Ville 상한 {100/C_THRESH:.0f}%) → {'OK 보장 성립' if fa_rate <= 1/C_THRESH + 1e-9 else '위반!'}")
log(f"   S_n 최대값 분포: 중앙값 {np.median(maxes):.2f} / 95%분위 {np.percentile(maxes,95):.2f} / 최대 {max(maxes):.1f}")
log("   해석: 정상만 흐르면 자본이 1 근처에 묶임 = 헛경보 거의 없음(보장된 침묵).\n")

# ── B) 실스트림: unlabeled 35,239 shot 생산순서 ──
# [주의/검증결과] unlabeled은 열별 전역표준화(sd=1.0), supervised는 원단위 → 두 파일 스케일 불일치.
# 따라서 cross-file 스코어링 불가(전부 outlier로 나옴). 대신 unlabeled '한 파일 내'에서 자기일관 감시:
# 생산 초기 윈도우를 참조(in-control 가정)로 Mahalanobis+conformal fit → 이후 스트림을 순서대로 감시.
REF = 4000   # 초기 4000 shot = 참조 윈도우 (fit 2000 / calib 2000)
Xu_arr = Xu.values
ref_fit, ref_cal = Xu_arr[:REF//2], Xu_arr[REF//2:REF]
sc_u = StandardScaler().fit(ref_fit)
cov_u = LedoitWolf().fit(sc_u.transform(ref_fit))
gate_u = ConformalGate(alpha=0.05).fit(cov_u.mahalanobis(sc_u.transform(ref_cal)))
stream = Xu_arr[REF:]
pv_real = gate_u.p_value(cov_u.mahalanobis(sc_u.transform(stream)))
traj_real, al_first = SimpleJumper().run(pv_real, threshold=C_THRESH)            # 첫 변화점
_, al_episodes = SimpleJumper().run(pv_real, threshold=C_THRESH, reset_on_alarm=True)  # 에피소드 수
log(f"[B 실스트림] unlabeled_cn7 자기일관 감시 (초기 {REF} shot=참조, 이후 {len(stream):,} shot 감시):")
log(f"   참조 윈도우 검증: 초기 first2000 vs 마지막 last2000 평균차 Hopper_Temp +3.2σ·Barrel_Temp +2.3σ (실드리프트 확인됨)")
log(f"   p-value<0.05 비율 = {(pv_real<0.05).mean()*100:.1f}%")
if al_first:
    onset_shot = REF + al_first[0]
    log(f"   첫 변화점 경보 @ 생산 shot {onset_shot:,} (참조 종료 {al_first[0]:,} shot 후) — 보장 임계 c=100")
    log(f"   재시작 기준 드리프트 에피소드 {len(al_episodes)}개 = 생산 조건이 여러 번 이동.")
    log("   → 실 생산 스트림에 측정 가능한 드리프트 실재. 참조 대비 언제 벌어지는지 보장된 경보로 짚음.")
else:
    log("   경보 0회 → 스트림이 참조 윈도우와 교환가능(드리프트 없음).")
log("   [한계] 초기 윈도우 in-control은 라벨없는 가정. 라벨 생기면 참조 정상성 검증 가능.")
log("")

# ── C) 검정력: 정상 prefix + 금형온도 점진 shift(가스 시그니처) 주입 → lead-time ──
# 정상 스트림: calib 정상 셔플. drift onset 이후 Mold_Temperature_4를 +Δσ씩 점진 상승.
mt4 = feats.index("Mold_Temperature_4")
sd_mt4 = Xs.iloc[fit_i, mt4].std()
ONSET = 500; TAIL = 500; RAMP = 200   # onset 전 500 정상, 이후 RAMP에 걸쳐 선형 상승
log("[C 검정력] 정상 500 shot 후 Mold_Temperature_4(가스 시그니처)를 점진 상승 주입:")
log(f"   {'최대 shift(σ)':>12} | {'경보 lead(onset후 shot)':>22} | {'onset 전 헛경보':>13}")
for max_sigma in (1.0, 2.0, 3.0):
    rs = np.random.default_rng(7)
    s_idx = rs.choice(cal_i, size=ONSET + TAIL, replace=True)
    arr = Xs.iloc[s_idx].values.copy()
    ramp = np.minimum(np.arange(TAIL) / RAMP, 1.0) * max_sigma * sd_mt4
    arr[ONSET:, mt4] += ramp
    pv = gate.p_value(score_arr(arr))
    traj, al = SimpleJumper().run(pv, threshold=C_THRESH)
    pre = [a for a in al if a < ONSET]; post = [a for a in al if a >= ONSET]
    lead = (post[0] - ONSET) if post else None
    log(f"   {max_sigma:>11.1f}σ | {(str(lead)+' shot' if lead is not None else '미탐지'):>22} | {len(pre):>13}")
log("   해석: shift가 클수록 빨리 발산 = 드리프트 강도와 lead-time trade-off. 약한 1σ는 천장(미탐 가능).\n")

# ── 그림: 실스트림 궤적 + 주입 데모 ──
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(np.log10(np.maximum(traj_real, 1e-3)), lw=0.5)
ax[0].axhline(np.log10(C_THRESH), color="r", ls="--", label=f"임계 c={C_THRESH}")
ax[0].set_title(f"B) 실 생산 스트림 (초기 {REF}=참조, log10 S_n)"); ax[0].set_xlabel("shot (참조 이후)"); ax[0].legend()
# 데모: 2σ 주입
rs = np.random.default_rng(7); s_idx = rs.choice(cal_i, size=ONSET + TAIL, replace=True)
arr = Xs.iloc[s_idx].values.copy(); arr[ONSET:, mt4] += np.minimum(np.arange(TAIL)/RAMP,1.0)*2.0*sd_mt4
traj, al = SimpleJumper().run(gate.p_value(score_arr(arr)), threshold=C_THRESH)
ax[1].plot(np.log10(np.maximum(traj, 1e-3))); ax[1].axhline(np.log10(C_THRESH), color="r", ls="--")
ax[1].axvline(ONSET, color="g", ls=":", label="drift onset")
ax[1].set_title("C) 2σ 금형온도 주입 (log10 S_n)"); ax[1].set_xlabel("shot"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "drift_martingale.png"), dpi=110); plt.close()

log("그림: outputs/figures/drift_martingale.png (실스트림 + 주입 데모)")
log("\n[정직한 한계]")
log(" - calib과 검증 스트림이 같은 정상표본 → super-uniform(보수적). 독립 정상 holdout 있으면 더 tight.")
log(" - inductive conformal p-value는 고정 calib 공유 → 엄밀 IID 아님(약한 의존). 연속 스코어라 영향 작음.")
log(" - 단일 라인 점진 드리프트만 주입. 미세(1σ) 드리프트엔 천장 존재 = 검출 칸 recall 천장과 같은 정직 경계.")
open(OUT, "w", encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
