"""18_demo_loo.py — 대시보드 데모 결함의 in-sample 누설 정량화 + LOO 수정 검증.

지도 모드 데모에서 결함 shot은 분류기 학습에 들어간다(demo_core.build의 Xdef = 가스+미성형 전부).
그래서 대시보드가 그 결함을 진단하면 in-sample 채점 = per-shot 확신 과대평가(누설).
수정: service.diagnose 가 그 shot 을 뺀 분류기로 재적합(LOO 교차적합)해 정직한 out-of-sample
진단을 낸다(eda/16 의 recall 측정과 같은 방법론). 비지도(Mahalanobis)는 결함 미학습 → 누설 없음.

이 스크립트는 두 경로(전체 학습 vs 제외 학습)의 conformal p-value/검출을 나란히 측정해
누설이 실재했고(평균 p 13배·검출 100→95%) LOO 로 닫혔음을 보인다.
실행: python eda/18_demo_loo.py   산출: outputs/18_demo_loo.txt
"""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import demo_core as core

OUT = os.path.join(ROOT, "outputs", "18_demo_loo.txt")
L = []; log = lambda s="": L.append(str(s))

det_full, _, X, reason, _ = core.build("supervised")           # 전체 결함 학습(데모 기존 = 누설)
didx = np.where(np.isin(reason, ['가스', '미성형']))[0]         # 데모로 노출되는 진짜결함 19

log("=== 대시보드 데모 결함: in-sample(전체 학습) vs LOO(제외) — 지도 게이트 ===")
log(f"결함 {len(didx)}개(가스13+미성형6) · 정상 split 고정(fit45/calib35/demo20)\n")
log(f"{'mode':7} {'p_insample':>11} {'p_LOO':>9} {'anom_in':>8} {'anom_LOO':>9}")

pf, pl, in_anom, loo_anom = [], [], 0, 0
for i in didx:
    df = det_full.diagnose(X.iloc[i])
    det_loo, _, _, _, _ = core.build("supervised", exclude_idx=int(i))    # 자기 제외 재적합
    dl = det_loo.diagnose(X.iloc[i])
    pf.append(df['p_value']); pl.append(dl['p_value'])
    in_anom += df['is_anomaly']; loo_anom += dl['is_anomaly']
    log(f"{str(reason[i]):7} {df['p_value']:>11.4f} {dl['p_value']:>9.4f} "
        f"{str(df['is_anomaly']):>8} {str(dl['is_anomaly']):>9}")

n = len(didx)
log(f"\n평균 p:  in-sample={np.mean(pf):.4f}  LOO={np.mean(pl):.4f}  (×{np.mean(pl)/np.mean(pf):.0f})")
log(f"검출(이상):  in-sample={in_anom}/{n} ({in_anom/n:.0%})  LOO={loo_anom}/{n} ({loo_anom/n:.0%})")
log("\n[해석]")
log(" - in-sample 평균 p 가 LOO 대비 한 자릿수 작음 = 데모 결함이 학습에 들어가 확신이 부풀려져 있었음.")
log(" - LOO 에서 '이상'→'기권'으로 뒤집힌 결함 = 그 shot 이 학습에 있을 때만 잡혔던 누설 검출.")
log(" - 수정 후 대시보드 지도 진단 = out-of-sample = eda/16 recall(89%)과 같은 정직한 기준.")
log(" - 데모 LOO 검출율과 eda/16 보고 recall 의 소차이는 정상 split/설정 차이(19표본 분산 범위).")
open(OUT, "w", encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
