"""02_separability.py — 타당성 게이트: 진짜 결함(가스/미성형)이 공정변수에서
정상과 구분되는가? 정상 분포로 표준화 후 결함군의 평균 z-편차(효과크기)를 본다.
실행: python eda/02_separability.py
산출: outputs/02_separability.txt
"""
import os, pandas as pd, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "02_separability.txt")
L = []
def log(s=""): L.append(str(s))

df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()

meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME',
        'EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce')
nunique = X.nunique()
dead = nunique[nunique <= 1].index.tolist()
live = [c for c in feats if c not in dead]
log(f"CN7 rows={len(cn7)}  raw feats={len(feats)}  live={len(live)}  missing={int(X[live].isna().sum().sum())}")
log(f"dead(상수) dropped: {dead}")

normal = cn7['PassOrFail']=='Y'
groups = {'가스': cn7['Reason']=='가스', '미성형': cn7['Reason']=='미성형',
          '초기허용불량(cold-start)': cn7['Reason']=='초기허용불량'}
log(f"\nnormal={int(normal.sum())}  " + "  ".join(f"{k}={int(v.sum())}" for k,v in groups.items()))

mu, sd = X[live][normal].mean(), X[live][normal].std().replace(0, np.nan)
Z = (X[live] - mu) / sd
for name, mask in groups.items():
    z = Z[mask].mean().dropna()
    z = z.reindex(z.abs().sort_values(ascending=False).index)
    log(f"\n[{name}] 정상 대비 평균 z-편차 top6:")
    for c in z.head(6).index:
        log(f"   {c:<26} z={z[c]:+.2f}")

log("\n분리도 sanity:")
for nm in ['가스','미성형']:
    m = cn7['Reason']==nm
    mz = Z[m].abs().max(axis=1)
    log(f"   {nm}: 결함 shot 최대|z| 중앙값={mz.median():.1f} 최소={mz.min():.1f}")
log(f"   정상 shot 중 |z|>3 변수 보유 비율: {(Z[normal].abs()>3).any(axis=1).mean()*100:.1f}%  (오탐 하한)")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT)
