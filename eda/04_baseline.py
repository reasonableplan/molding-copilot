"""04_baseline.py — 2단계 베이스라인.
(A) PCA 선형 오토인코더: 정상으로 fit → 재구성오차 = anomaly score (레이어2 grounding 솔기).
(B) 지도 로지스틱(클래스밸런싱): 발표 벤치마크 궤도 확인용.
평가는 (가스+미성형=진짜) / (초기허용불량=쉬운셋) / (전체) 로 분리 리포트.
실행: python eda/04_baseline.py   산출: outputs/04_baseline.txt, outputs/figures/anomaly_scores.png
"""
import os, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
rng = np.random.default_rng(42)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT,"data","Dataset_Molding","dataset")
FIG = os.path.join(ROOT,"outputs","figures"); os.makedirs(FIG, exist_ok=True)
OUT = os.path.join(ROOT,"outputs","04_baseline.txt")
L=[]; log=lambda s="": L.append(str(s))

df = pd.read_csv(os.path.join(D,"labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7',na=False)].copy()
meta=['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats=[c for c in cn7.columns if c not in meta]
X=cn7[feats].apply(pd.to_numeric,errors='coerce')
live=[c for c in feats if X[c].nunique()>1]; X=X[live].fillna(X[live].median())
reason=cn7['Reason']; is_normal=(cn7['PassOrFail']=='Y').values
real=reason.isin(['가스','미성형']).values   # 진짜 결함 19
easy=(reason=='초기허용불량').values          # cold-start 20
log(f"CN7={len(cn7)}  feats={len(live)}  normal={is_normal.sum()}  진짜(가스+미성형)={real.sum()}  초기허용={easy.sum()}")

# ---- (A) PCA autoencoder: fit on TRAIN-normal only ----
nidx=np.where(is_normal)[0]; rng.shuffle(nidx)
cut=int(len(nidx)*0.7); tr=nidx[:cut]; te_norm=nidx[cut:]
sc=StandardScaler().fit(X.iloc[tr])
Xs=sc.transform(X)
k=10
pca=PCA(n_components=k).fit(Xs[tr])
recon=pca.inverse_transform(pca.transform(Xs))
score=np.sum((Xs-recon)**2,axis=1)   # 재구성오차 = anomaly score
log(f"\n(A) PCA-AE: k={k}, 설명분산={pca.explained_variance_ratio_.sum()*100:.1f}%")

def evalset(name, pos_mask):
    y=np.zeros(len(X),bool); y[te_norm]=False; y[pos_mask]=True
    keep=np.zeros(len(X),bool); keep[te_norm]=True; keep[pos_mask]=True
    yy=y[keep]; ss=score[keep]
    ap=average_precision_score(yy,ss); auc=roc_auc_score(yy,ss)
    thr=np.percentile(score[te_norm],95)            # FPR≈5%
    rec=(score[pos_mask]>=thr).mean()
    log(f"   [{name:<16}] PR-AUC={ap:.3f}  ROC-AUC={auc:.3f}  recall@FPR5%={rec:.2f}  (pos={pos_mask.sum()})")
log("평가(테스트-정상 vs 결함):")
evalset("진짜 가스+미성형", real)
evalset("쉬운 초기허용", easy)
evalset("전체 결함", is_normal==False)

# blind case 확인
log(f"   미성형 중 score가 테스트-정상 중앙값 이하인 수: "
    f"{int((score[(reason=='미성형').values] < np.median(score[te_norm])).sum())}/{int((reason=='미성형').sum())} (= 안 보이는 맹점)")

# ---- (B) supervised LR (class-balanced), 전체 39 결함 기준, 5-fold ----
y_all=(~is_normal).astype(int)
cv=StratifiedKFold(5,shuffle=True,random_state=42)
proba=cross_val_predict(LogisticRegression(max_iter=2000,class_weight='balanced'),
                        Xs,y_all,cv=cv,method='predict_proba')[:,1]
log(f"\n(B) 지도 LR(balanced) 5-fold: PR-AUC={average_precision_score(y_all,proba):.3f}  "
    f"ROC-AUC={roc_auc_score(y_all,proba):.3f}  (발표 SL-CBL: F1~0.97/AUC~0.99, 모델 다름)")

# ---- score 분포 그림 ----
fig,ax=plt.subplots(figsize=(9,5))
ax.hist(np.log10(score[te_norm]+1e-9),bins=50,color='#cccccc',density=True,label='정상(test)')
for g,c in [('가스','#d62728'),('미성형','#1f77b4'),('초기허용불량','#ff7f0e')]:
    s=score[(reason==g).values]
    for v in s: ax.axvline(np.log10(v+1e-9),color=c,alpha=0.7,lw=1.2)
ax.axvline(np.log10(np.percentile(score[te_norm],95)+1e-9),color='k',ls='--',label='FPR5% 임계')
ax.set_xlabel('log10(재구성오차)'); ax.set_yticks([]); ax.set_title('PCA-AE anomaly score 분포'); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG,'anomaly_scores.png'),dpi=110); plt.close(fig)

open(OUT,"w",encoding="utf-8").write("\n".join(L)); print("wrote",OUT)
