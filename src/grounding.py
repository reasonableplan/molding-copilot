"""grounding.py — 레이어2: grounded 결함 설명 엔진.

정상 분포를 학습한 뒤, 한 shot에 대해 '정상에서 벗어난 변수'를 근거로 인용한다.
임계를 넘는 변수가 없으면 지어내지 않고 '근거 불충분(모른다)'을 반환한다.
인용된 변수를 도메인 시그니처와 대조해 결함모드를 추정한다 (가스=과열, 미성형=충전부족).
"""
import numpy as np, pandas as pd

# 검증된 분리도(step1/1.5)에 기반한 도메인 시그니처 — 인용 변수 → 결함모드
SIGNATURE = {
    '가스':   ['Mold_Temperature_3', 'Mold_Temperature_4', 'Hopper_Temperature',
              'Barrel_Temperature_1', 'Barrel_Temperature_2', 'Barrel_Temperature_3'],
    '미성형': ['Max_Injection_Speed', 'Filling_Time', 'Injection_Time',
              'Max_Injection_Pressure', 'Average_Screw_RPM'],
}
_VAR2MODE = {v: m for m, vs in SIGNATURE.items() for v in vs}

# z-인용 임계 — 값이 다른 건 우연이 아니라 의도된 차이(게이트 有/無):
#   · GATED(2.0): conformal 게이트가 이미 이상판정함 → z는 '어느 변수를 보여줄까'의 인용 임계(완화).  → src/detector.py
#   · STANDALONE(3.0): 게이트 없이 z만으로 이상판정 → z가 게이트 역할까지 함(보수적 3σ).            → GroundedExplainer(이 파일)
Z_CITE_GATED = 2.0
Z_CITE_STANDALONE = 3.0


def infer_mode(evidence: list) -> str | None:
    """인용 변수 → 결함모드 |z|가중 투표(가장 결정적 편차가 지배). 인용 없으면 None.
    detector(게이트 후)·GroundedExplainer(게이트 전) 공유 = 모드추정 로직 단일 출처."""
    w = {}
    for e in evidence:
        m = _VAR2MODE.get(e['var'])
        if m:
            w[m] = w.get(m, 0.0) + abs(e['z'])
    return max(w, key=w.get) if w else None


class GroundedExplainer:
    def __init__(self, z_threshold=Z_CITE_STANDALONE, top_k=3):
        self.t = z_threshold      # 이 |z| 미만이면 '모른다'(게이트無 standalone)
        self.k = top_k

    def fit(self, X_normal: pd.DataFrame):
        self.cols = list(X_normal.columns)
        self.mu = X_normal.mean()
        self.sd = X_normal.std().replace(0, np.nan)
        return self

    def explain(self, shot: pd.Series) -> dict:
        z = (shot[self.cols] - self.mu) / self.sd
        z = z.dropna()
        ranked = z.reindex(z.abs().sort_values(ascending=False).index)
        cited = ranked[ranked.abs() >= self.t].head(self.k)

        if len(cited) == 0:                       # ── 모른다 게이트
            return {'verdict': '근거불충분', 'mode': None, 'cited': [],
                    'max_abs_z': float(ranked.abs().max()),
                    'text': "근거 불충분: 측정 공정변수상 정상과 구분되지 않음 (지어내지 않음)."}

        evidence = []
        for v in cited.index:
            evidence.append({'var': v, 'observed': float(shot[v]),
                             'normal_mean': float(self.mu[v]), 'normal_sd': float(self.sd[v]),
                             'z': float(z[v])})
        mode = infer_mode(evidence)      # |z|가중 투표(detector와 공유)

        top = evidence[0]
        direction = '높음' if top['z'] > 0 else '낮음'
        text = (f"이상 판정. 근거: {top['var']} = {top['observed']:.1f} "
                f"(정상 {top['normal_mean']:.1f}±{top['normal_sd']:.1f}, {top['z']:+.1f}σ {direction})"
                + (f" → '{mode}' 패턴과 일치." if mode else " → 도메인 매핑 없음."))
        return {'verdict': '이상', 'mode': mode, 'cited': [e['var'] for e in evidence],
                'evidence': evidence, 'max_abs_z': float(ranked.abs().max()), 'text': text}
