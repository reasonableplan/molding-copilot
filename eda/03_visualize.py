"""03_visualize.py — 정상 분포 위에 결함(가스/미성형/초기허용불량)을 얹어 눈으로 본다.
실행: python eda/03_visualize.py
산출: outputs/figures/{hist,scatter,pca}.png
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
FIG = os.path.join(ROOT, "outputs", "figures"); os.makedirs(FIG, exist_ok=True)

df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME',
        'EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce')
live = [c for c in feats if X[c].nunique() > 1]
X = X[live]

# class label
cls = pd.Series('정상', index=cn7.index)
cls[cn7['Reason']=='가스'] = '가스'
cls[cn7['Reason']=='미성형'] = '미성형'
cls[cn7['Reason']=='초기허용불량'] = '초기허용불량'
styles = {'정상':('#bbbbbb',6,0.25,'o'), '가스':('#d62728',70,0.9,'X'),
          '미성형':('#1f77b4',90,0.95,'P'), '초기허용불량':('#ff7f0e',45,0.7,'s')}

# ---- Fig 1: histograms of top separating features ----
keyvars = ['Max_Injection_Speed','Mold_Temperature_4','Mold_Temperature_3','Filling_Time']
fig, axes = plt.subplots(2,2, figsize=(12,8))
for ax,v in zip(axes.ravel(), keyvars):
    nv = X.loc[cls=='정상', v]
    ax.hist(nv, bins=60, color='#cccccc', label='정상', density=True)
    for g in ['가스','미성형','초기허용불량']:
        gv = X.loc[cls==g, v]
        for x in gv:
            ax.axvline(x, color=styles[g][0], alpha=0.7, lw=1.2)
    ax.set_title(v); ax.set_yticks([])
axes[0,0].legend()
fig.suptitle('정상 분포(회색) 위 결함 위치(세로선) — CN7', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG,'hist.png'), dpi=110); plt.close(fig)

# ---- Fig 2: 2D scatter on two domain axes ----
fig, ax = plt.subplots(figsize=(9,7))
for g in ['정상','초기허용불량','가스','미성형']:
    m = cls==g; c,s,a,mk = styles[g]
    ax.scatter(X.loc[m,'Max_Injection_Speed'], X.loc[m,'Mold_Temperature_4'],
               c=c, s=s, alpha=a, marker=mk, label=f'{g} ({int(m.sum())})',
               edgecolors='k' if g!='정상' else 'none', linewidths=0.4)
ax.set_xlabel('Max_Injection_Speed  (미성형 축)')
ax.set_ylabel('Mold_Temperature_4  (가스 축)')
ax.set_title('도메인 2축 산점도 — 결함이 정상 군집에서 벗어나는가')
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG,'scatter.png'), dpi=110); plt.close(fig)

# ---- Fig 3: PCA on all 24 features (standardized on normal) ----
mu, sd = X[cls=='정상'].mean(), X[cls=='정상'].std().replace(0,np.nan)
Z = ((X - mu)/sd).fillna(0)
p = PCA(n_components=2).fit(Z[cls=='정상'])
P = p.transform(Z)
fig, ax = plt.subplots(figsize=(9,7))
for g in ['정상','초기허용불량','가스','미성형']:
    m = (cls==g).values; c,s,a,mk = styles[g]
    ax.scatter(P[m,0], P[m,1], c=c, s=s, alpha=a, marker=mk, label=f'{g} ({int(m.sum())})',
               edgecolors='k' if g!='정상' else 'none', linewidths=0.4)
ev = p.explained_variance_ratio_
ax.set_xlabel(f'PC1 ({ev[0]*100:.0f}%)'); ax.set_ylabel(f'PC2 ({ev[1]*100:.0f}%)')
ax.set_title('24 공정변수 PCA (정상 기준 표준화)')
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG,'pca.png'), dpi=110); plt.close(fig)

print("saved:", os.listdir(FIG))
