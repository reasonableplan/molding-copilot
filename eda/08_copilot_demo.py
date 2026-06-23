"""08_copilot_demo.py — grounded copilot 실동작 데모 (Ollama 호출).
실행: python eda/08_copilot_demo.py   산출: outputs/08_copilot.txt
"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.detector import Detector
from src.copilot import Copilot

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "08_copilot.txt"); L = []; log = lambda s="": L.append(str(s))

df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce'); live = [c for c in feats if X[c].nunique() > 1]
X = X[live].fillna(X[live].median()); reason = cn7['Reason'].values; is_normal = (cn7['PassOrFail']=='Y').values

nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
a, b = int(len(nidx)*0.5), int(len(nidx)*0.75)
det = Detector(alpha=0.05, top_k=3).fit(X.iloc[nidx[:a]], X.iloc[nidx[a:b]])
cop = Copilot(det, model="gemma3:4b")

# 발화된 미성형 1건, 게이트된 가스 1건, 정상 1건
mis = [i for i in np.where(reason=='미성형')[0] if det.diagnose(X.iloc[i])['is_anomaly']][:1]
gas_gated = [i for i in np.where(reason=='가스')[0] if not det.diagnose(X.iloc[i])['is_anomaly']][:1]
samples = [(mis[0], "미성형(발화)"), (gas_gated[0], "가스(게이트)"), (int(nidx[b]), "정상")]

for i, tag in samples:
    r = cop.ask(X.iloc[i])
    log(f"\n=== [{tag}] 실제Reason={reason[i] if not is_normal[i] else '정상'} ===")
    log(f"진단: verdict={r['diagnosis']['verdict']} mode={r['diagnosis']['mode']} p={r['diagnosis']['p_value']:.3f}")
    log(f"copilot 답변:\n  {r['answer']}")

open(OUT,"w",encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
