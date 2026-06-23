"""05_grounding_eval.py — 레이어2 grounded 설명의 정직성 평가.
- 진짜 결함 19(가스13+미성형6)에 설명 생성 → 인용변수→추정모드가 실제 Reason과 일치하는가(적중률)
- '모른다' 게이트로 거른 수 / 정상 shot에 잘못 발화한 비율(정밀도 비용)
- naive 베이스라인(항상 다수결 모드 추측) 대비 우위
실행: python eda/05_grounding_eval.py   산출: outputs/05_grounding.txt
"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.grounding import GroundedExplainer

D = os.path.join(ROOT, "data", "Dataset_Molding", "dataset")
OUT = os.path.join(ROOT, "outputs", "05_grounding.txt")
L = []; log = lambda s="": L.append(str(s))

df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
feats = [c for c in cn7.columns if c not in meta]
X = cn7[feats].apply(pd.to_numeric, errors='coerce')
live = [c for c in feats if X[c].nunique() > 1]
X = X[live].fillna(X[live].median())

is_normal = (cn7['PassOrFail'] == 'Y').values
reason = cn7['Reason'].values
real_mask = np.isin(reason, ['가스', '미성형'])

# 정상 70%로 fit, 나머지 30%는 오발화 측정용
nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
cut = int(len(nidx)*0.7); tr, te_norm = nidx[:cut], nidx[cut:]
exp = GroundedExplainer(z_threshold=3.0, top_k=3).fit(X.iloc[tr])

# ── 진짜 결함 19개 평가
hits = gated = fired = 0
per = {'가스': [0,0,0], '미성형': [0,0,0]}   # [fired, hit, gated]
rows = []
for i in np.where(real_mask)[0]:
    r = exp.explain(X.iloc[i]); actual = reason[i]
    rows.append((actual, r))
    if r['verdict'] == '근거불충분':
        gated += 1; per[actual][2] += 1
    else:
        fired += 1; per[actual][0] += 1
        if r['mode'] == actual:
            hits += 1; per[actual][1] += 1

log(f"진짜 결함 19 (가스13+미성형6) 설명 평가:")
log(f"  발화(이상): {fired}  게이트(모른다): {gated}")
log(f"  grounded 적중(추정모드=실제Reason): {hits}/{fired} = {hits/max(fired,1)*100:.0f}% (발화건 기준)")
for m,(f,h,g) in per.items():
    log(f"    {m}: 발화{f} 적중{h} 게이트{g}")

# ── naive 베이스라인: per-shot 근거 없이 항상 다수결 모드('가스' 13>6) 추측
maj = '가스'; naive_hit = (reason[real_mask] == maj).sum()
log(f"\nnaive 베이스라인(항상 '{maj}' 추측, 근거 없음): {naive_hit}/19 = {naive_hit/19*100:.0f}%  (게이트 능력 없음)")

# ── 정상 오발화 (정밀도 비용)
fp = sum(exp.explain(X.iloc[i])['verdict'] == '이상' for i in te_norm)
log(f"\n정상 test {len(te_norm)}건 중 '이상' 오발화: {fp} = {fp/len(te_norm)*100:.1f}% (= 정밀도 비용, |z|>3 정상)")

# ── 샘플 설명 3종 (가스 / 미성형 / 모른다 게이트 사례)
log("\n--- 샘플 설명 ---")
shown = {'가스': False, '미성형': False, '게이트': False}
for actual, r in rows:
    key = '게이트' if r['verdict'] == '근거불충분' else actual
    if key in shown and not shown[key]:
        shown[key] = True
        log(f"[실제={actual}] {r['text']}")
    if all(shown.values()): break

open(OUT, "w", encoding="utf-8").write("\n".join(L)); print("wrote", OUT)
