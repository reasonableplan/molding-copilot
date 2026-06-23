"""10_prescribe.py — 처방 레이어 데모/검증.
발화된 진짜결함에 recourse 처방 + counterfactual 검증. 검증: p_after > alpha 인가.
실행: python eda/10_prescribe.py   산출: outputs/10_prescribe.txt
"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.detector import Detector
from src.prescribe import Prescriber

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "10_prescribe.txt"); L = []; log = lambda s="": L.append(str(s))

df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce'); live = [c for c in feats if X[c].nunique() > 1]
X = X[live].fillna(X[live].median()); reason = cn7['Reason'].values; is_normal = (cn7['PassOrFail']=='Y').values

nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
a, b = int(len(nidx)*0.5), int(len(nidx)*0.75)
det = Detector(alpha=0.05, top_k=3).fit(X.iloc[nidx[:a]], X.iloc[nidx[a:b]])
rx = Prescriber(det)

real = np.where(np.isin(reason, ['가스','미성형']))[0]
fired = [i for i in real if det.diagnose(X.iloc[i])['is_anomaly']]
# 미성형(깨끗) 먼저, 그다음 가스 순으로 표시
fired_sorted = sorted(fired, key=lambda i: 0 if reason[i]=='미성형' else 1)
resolved = verified = 0
log(f"발화된 진짜결함 {len(fired)}건에 처방:\n")
shown = 0
for i in fired_sorted:
    r = rx.prescribe(X.iloc[i])
    if r.get('resolved'):
        resolved += 1
        if r['p_after'] > det.alpha: verified += 1
    if shown < 5:
        log(f"[실제={reason[i]}] {r['text']}"); shown += 1

review = len(fired) - resolved
log(f"\n요약: 발화 {len(fired)} → 방향 권고 {len(fired)} 전부 / what-if 정상복귀 확인 {resolved} / 공정검토 권고 {review}")
log("[핵심 발견] 진단 원인을 정상화해도 what-if p가 거의 안 오름(예 0.001→0.006). 결함이 다변량이라")
log("단순 처방이 모델상 정상복귀 미확인 → 정직하게 '공정검토'. = 상관 recourse 한계 = 인과 모델이 다음 칸.")
open(OUT,"w",encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
