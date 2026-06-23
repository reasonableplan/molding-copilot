"""demo_core.py — ML 파이프라인 접근 코어 (웹 백엔드 webapp/api/service.py 가 재사용).

데이터 로드 + detector/prescriber 구성(지도/비지도 게이트). streamlit-무관, 단독 테스트 가능.
"""
import os
import numpy as np
import pandas as pd
from src.detector import Detector
from src.prescribe import Prescriber

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Dataset_Molding", "dataset")

# 검증된 헤드라인 숫자(outputs/ 에서 측정 — 표시는 상수로 캐시). 각 출처 eda 스크립트 명시.
TRUST = {
    'recall_unsup': 58, 'recall_sup': 89,                       # eda/16
    'recall_ci': (36, 77),                                      # eda/14 부트스트랩
    'cw_ours': 7.5, 'cw_vanilla': 67.5, 'cw_diff_ci': (43, 77), # eda/09,14
    'fpr_guarantee': 4.94, 'alpha': 5,                          # eda/06 (K=300)
    'rg3_fpr': 4.9, 'rg3_recall': 12.5,                         # eda/15
}


def load_cn7():
    df = pd.read_csv(os.path.join(D, "labeled_data.csv"))
    cn7 = df[df['PART_NAME'].str.startswith('CN7', na=False)].copy()
    meta = ['_id','TimeStamp','PART_FACT_PLAN_DATE','PART_FACT_SERIAL','PART_NAME','EQUIP_CD','EQUIP_NAME','PassOrFail','Reason']
    feats = [c for c in cn7.columns if c not in meta]
    X = cn7[feats].apply(pd.to_numeric, errors='coerce')
    live = [c for c in feats if X[c].nunique() > 1]
    X = X[live].fillna(X[live].median()).reset_index(drop=True)
    reason = cn7['Reason'].fillna('정상').values
    is_normal = (cn7['PassOrFail'] == 'Y').values
    return X, reason, is_normal


def build(gate_mode='supervised', exclude_idx=None):
    """gate_mode: 'supervised'(지도 GBM) | 'mahalanobis'(비지도).
    정상 split: fit 45% / calib 35% / demo 20%(held-out — 데모 정상은 학습 미포함=누출 없음).
    exclude_idx: 지도 모드에서 이 결함 shot을 학습에서 제외(LOO 교차적합 — 데모 결함 in-sample 누설 차단)."""
    X, reason, is_normal = load_cn7()
    nidx = np.where(is_normal)[0]; rng = np.random.default_rng(42); rng.shuffle(nidx)
    a, b = int(len(nidx)*0.45), int(len(nidx)*0.80)
    Xfit, Xcal = X.iloc[nidx[:a]], X.iloc[nidx[a:b]]
    demo_normal = list(nidx[b:])
    if gate_mode == 'supervised':
        didx = np.where(np.isin(reason, ['가스', '미성형']))[0]
        if exclude_idx is not None:
            didx = didx[didx != exclude_idx]      # 진단 대상 결함은 학습서 제외
        Xdef = X.iloc[didx]
    else:
        Xdef = None
    det = Detector().fit(Xfit, Xcal, X_defects=Xdef)
    return det, Prescriber(det), X, reason, demo_normal


def sample_indices(reason, demo_normal):
    """데모용 인덱스: 가스/미성형 결함 + held-out 정상."""
    out = {m: list(np.where(reason == m)[0]) for m in ['가스', '미성형']}
    out['정상'] = list(demo_normal)
    return out


if __name__ == "__main__":
    for gm in ['supervised', 'mahalanobis']:
        det, presc, X, reason, demo_normal = build(gm)
        idx = sample_indices(reason, demo_normal)
        di = idx['미성형'][0]; ni = idx['정상'][0]
        dd = det.diagnose(X.iloc[di]); nn = det.diagnose(X.iloc[ni])
        print(f"[{gm:11}] 결함 anom={dd['is_anomaly']} mode={dd['mode']} | 정상 anom={nn['is_anomaly']}")
    print("demo_core 스모크 OK")
