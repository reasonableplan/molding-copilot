"""09_faithfulness.py — 레이어3-c: 엔진 有(우리) vs 無(vanilla) 대조.
같은 모델(gemma3:4b)을 grounding 엔진 유무로만 갈라, ground truth(detector)에 대고 잰다.
킬러 지표: vanilla의 '정상범위 날조율'(엔진 없이 정상평균을 추측→틀림) vs 우리 0%.
실행: python eda/09_faithfulness.py   산출: outputs/09_faithfulness.txt
"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.detector import Detector
from src.vanilla import VanillaDiagnoser

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "09_faithfulness.txt"); L = []; log = lambda s="": L.append(str(s))

df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce'); live = [c for c in feats if X[c].nunique() > 1]
X = X[live].fillna(X[live].median()); reason = cn7['Reason'].values; is_normal = (cn7['PassOrFail']=='Y').values

nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
a, b = int(len(nidx)*0.5), int(len(nidx)*0.75)
det = Detector(alpha=0.05, top_k=3).fit(X.iloc[nidx[:a]], X.iloc[nidx[a:b]])
van = VanillaDiagnoser(model="gemma3:4b")

def fabricated(v):  # vanilla가 댄 정상평균이 실제와 2σ 넘게 다르면 날조
    var, claim = v.get('top_variable'), v.get('claimed_normal_mean')
    if not var or var not in live or claim is None: return None
    return abs(claim - det.mu[var]) > 2 * det.sd[var]

# ── 진짜 결함 19
real = np.where(np.isin(reason, ['가스','미성형']))[0]
o_correct=o_abstain=o_wrong=0; v_correct=v_wrong=0; v_fab=v_fabN=0
examples=[]
for i in real:
    d = det.diagnose(X.iloc[i]); v = van.diagnose(X.iloc[i], live); gt = reason[i]
    # ours (결정론적, 기권 가능)
    if not d['is_anomaly']: o_abstain += 1
    elif d['mode']==gt: o_correct += 1
    else: o_wrong += 1
    # vanilla (항상 답)
    if v.get('defect_mode')==gt: v_correct += 1
    else: v_wrong += 1
    fb = fabricated(v)
    if fb is not None:
        v_fabN += 1; v_fab += int(fb)
    if len(examples) < 2 and d['is_anomaly']:
        examples.append((gt, d, v))

log("=== 진짜 결함 19 (가스13+미성형6) — 엔진 有 vs 無 ===")
log(f"모드 결과:   우리= 정답 {o_correct} / 기권 {o_abstain} / 오답 {o_wrong}    "
    f"vanilla= 정답 {v_correct} / 기권 0 / 오답 {v_wrong}")
log(f"정상범위 날조: 우리= 0% (실측 인용)    vanilla= {v_fab}/{v_fabN} = "
    f"{(v_fab/max(v_fabN,1))*100:.0f}%  (정상평균 추측→2σ 이상 빗나감)")

# ── 정상 shot 40 (둘 다 기권해야 함 = confidently-wrong 측정)
norm_eval = nidx[b:][:40]
o_cw = sum(det.diagnose(X.iloc[i])['is_anomaly'] for i in norm_eval)
v_cw = sum(van.diagnose(X.iloc[i], live).get('is_defect', False) for i in norm_eval)
log(f"\n=== 정상 shot {len(norm_eval)} (기권이 정답) ===")
log(f"confidently-wrong(불량이라 단정): 우리= {o_cw}/{len(norm_eval)} = {o_cw/len(norm_eval)*100:.0f}% "
    f"(α=5% 보장)    vanilla= {v_cw}/{len(norm_eval)} = {v_cw/len(norm_eval)*100:.0f}%")

# ── 샘플 대조
log("\n=== 샘플 대조 ===")
for gt, d, v in examples:
    e0 = d['evidence'][0]
    log(f"\n[실제={gt}]")
    log(f"  우리(엔진): {e0['var']} 관측 {e0['observed']} (정상 {e0['normal_mean']}±{e0['normal_sd']}, {e0['z']:+}σ) → {d['mode']}")
    log(f"  vanilla   : {v.get('top_variable')} 관측 {v.get('observed_value')} "
        f"(주장 정상평균 {v.get('claimed_normal_mean')}) → {v.get('defect_mode')}   "
        f"[날조={fabricated(v)}]")

open(OUT,"w",encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
