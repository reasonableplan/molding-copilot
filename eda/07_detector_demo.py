"""07_detector_demo.py — Detector(게이트+근거) 결정론적 검증. LLM 없음.
실행: python eda/07_detector_demo.py   산출: outputs/07_detector.txt
"""
import os, sys, json, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.detector import Detector

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "07_detector.txt"); L = []; log = lambda s="": L.append(str(s))

df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce'); live = [c for c in feats if X[c].nunique() > 1]
X = X[live].fillna(X[live].median()); reason = cn7['Reason'].values; is_normal = (cn7['PassOrFail']=='Y').values

nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
a, b = int(len(nidx)*0.5), int(len(nidx)*0.75)
det = Detector(alpha=0.05, top_k=3).fit(X.iloc[nidx[:a]], X.iloc[nidx[a:b]])

def show(i, tag):
    r = det.diagnose(X.iloc[i])
    log(f"\n[{tag}] verdict={r['verdict']} mode={r['mode']} p={r['p_value']:.3f} (α={r['alpha']})")
    for e in r['evidence']:
        log(f"    {e['var']:<24} 관측 {e['observed']} (정상 {e['normal_mean']}±{e['normal_sd']}, {e['z']:+}σ)")

# 가스 2건, 미성형 2건, test-정상 2건
for i in np.where(reason=='가스')[0][:2]: show(i, "실제=가스")
for i in np.where(reason=='미성형')[0][:2]: show(i, "실제=미성형")
for i in nidx[b:][:2]: show(i, "실제=정상")

# 전체 진짜결함에 대한 모드 추정 요약(게이트 통과분)
real = np.where(np.isin(reason, ['가스','미성형']))[0]
fired = [det.diagnose(X.iloc[i]) for i in real]
nf = sum(r['is_anomaly'] for r in fired)
hit = sum(r['is_anomaly'] and r['mode']==reason[i] for i, r in zip(real, fired))
log(f"\n요약: 진짜결함 {len(real)} → 발화 {nf}, 게이트 {len(real)-nf}, 모드적중 {hit}/{nf}")

open(OUT,"w",encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
