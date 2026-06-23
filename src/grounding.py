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


class GroundedExplainer:
    def __init__(self, z_threshold=3.0, top_k=3):
        self.t = z_threshold      # 이 |z| 미만이면 '모른다'
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
        # 인용 변수 다수결로 결함모드 추정
        votes = [_VAR2MODE.get(e['var']) for e in evidence if _VAR2MODE.get(e['var'])]
        mode = max(set(votes), key=votes.count) if votes else None

        top = evidence[0]
        direction = '높음' if top['z'] > 0 else '낮음'
        text = (f"이상 판정. 근거: {top['var']} = {top['observed']:.1f} "
                f"(정상 {top['normal_mean']:.1f}±{top['normal_sd']:.1f}, {top['z']:+.1f}σ {direction})"
                + (f" → '{mode}' 패턴과 일치." if mode else " → 도메인 매핑 없음."))
        return {'verdict': '이상', 'mode': mode, 'cited': [e['var'] for e in evidence],
                'evidence': evidence, 'max_abs_z': float(ranked.abs().max()), 'text': text}
