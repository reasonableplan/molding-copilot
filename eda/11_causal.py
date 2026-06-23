"""11_causal.py — 인과 탐색(DirectLiNGAM)으로 공정변수 인과순서 = lever vs symptom.
정상 공정데이터에 DirectLiNGAM → causal_order_(상류=원인) + adjacency(누가 누구를 일으키나).
진단 원인변수(금형온도/사출속도)가 상류 lever인지 하류 symptom인지 확인.
[가정] 선형·비가우스·은닉교란 없음 — 위반 가능, 결과는 가설로 취급.
실행: python eda/11_causal.py   산출: outputs/11_causal.txt
"""
import os, sys, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
import lingam
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "11_causal.txt"); L = []; log = lambda s="": L.append(str(s))

df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce'); live = [c for c in feats if X[c].nunique() > 1]
X = X[live].fillna(X[live].median()); is_normal = (cn7['PassOrFail']=='Y').values

# 정상 fit 셋 (detector와 동일 split)으로 인과 학습
nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
fit_i = nidx[:int(len(nidx)*0.5)]
Xn = pd.DataFrame(StandardScaler().fit_transform(X.iloc[fit_i]), columns=live)

model = lingam.DirectLiNGAM(random_state=42)
model.fit(Xn)
order = [live[i] for i in model.causal_order_]
B = model.adjacency_matrix_      # B[i,j] = j가 i에 주는 직접효과

log(f"DirectLiNGAM 인과순서 (상류=원인 → 하류=결과), n={len(fit_i)}:")
for r, v in enumerate(order):
    log(f"  {r:2d}. {v}")

def parents(var, topn=3):
    i = live.index(var)
    eff = [(live[j], B[i, j]) for j in range(len(live)) if abs(B[i, j]) > 0.1 and j != i]
    eff.sort(key=lambda x: -abs(x[1]))
    return eff[:topn]

log("\n진단 원인변수의 인과 위치 (parents = 이 변수를 일으키는 상류):")
for v in ['Mold_Temperature_4', 'Mold_Temperature_3', 'Max_Injection_Speed']:
    pos = order.index(v); par = parents(v)
    role = "상류 lever(원인 후보)" if not par else "하류 symptom(상류에 의해 야기됨)"
    log(f"  [{v}] 순서 {pos}/{len(live)-1} → {role}")
    for pv, w in par:
        log(f"        ← {pv} (효과 {w:+.2f})")

log("\n[가정/한계] 선형·비가우스·은닉교란 없음 가정. 단일 라인 데이터라 setpoint 변동 적으면 식별 약함.")
log("결과는 '가설적 인과'로 취급 — 도메인/실험으로 검증 필요.")
open(OUT,"w",encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
