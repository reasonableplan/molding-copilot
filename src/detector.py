"""detector.py — 결정론적 진단 코어 (LLM 없이 동작).
[conformal 게이트] 이상/모른다 판정 + [변수 근거] 어느 공정변수가 정상서 벗어났나.
copilot은 이 diagnose() 출력을 자연어로 표현만 한다 (숫자 생성 X = hallucination 차단).
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from src.conformal import ConformalGate
from src.grounding import SIGNATURE, _VAR2MODE


class Detector:
    def __init__(self, alpha=0.05, top_k=3, z_threshold=2.0):
        self.alpha, self.k, self.zt = alpha, top_k, z_threshold

    def fit(self, X_fit: pd.DataFrame, X_calib: pd.DataFrame):
        """X_fit: 정상(모델 적합), X_calib: 정상(conformal 보정). 분리 필수."""
        self.cols = list(X_fit.columns)
        self.scaler = StandardScaler().fit(X_fit)
        self.cov = LedoitWolf().fit(self.scaler.transform(X_fit))
        self.mu = X_fit.mean(); self.sd = X_fit.std().replace(0, np.nan)
        self.gate = ConformalGate(self.alpha).fit(self._score(X_calib))
        return self

    def _score(self, X):
        return self.cov.mahalanobis(self.scaler.transform(X[self.cols]))

    def diagnose(self, shot: pd.Series) -> dict:
        s = float(self._score(shot[self.cols].to_frame().T))
        p = float(self.gate.p_value(s))
        z = ((shot[self.cols] - self.mu) / self.sd).dropna()
        ranked = z.reindex(z.abs().sort_values(ascending=False).index)

        if p > self.alpha:                          # ── conformal 게이트: 모른다
            return {'verdict': '근거불충분', 'is_anomaly': False, 'p_value': p,
                    'alpha': self.alpha, 'mahalanobis': s, 'mode': None, 'evidence': []}

        cited = ranked[ranked.abs() >= self.zt].head(self.k)
        evidence = [{'var': v, 'observed': round(float(shot[v]), 2),
                     'normal_mean': round(float(self.mu[v]), 2),
                     'normal_sd': round(float(self.sd[v]), 2), 'z': round(float(z[v]), 2)}
                    for v in cited.index]
        # 모드 추정: 단순 다수결 아닌 |z| 가중 (가장 결정적인 편차가 지배)
        wmode = {}
        for e in evidence:
            m = _VAR2MODE.get(e['var'])
            if m:
                wmode[m] = wmode.get(m, 0.0) + abs(e['z'])
        mode = max(wmode, key=wmode.get) if wmode else None
        return {'verdict': '이상', 'is_anomaly': True, 'p_value': p, 'alpha': self.alpha,
                'mahalanobis': s, 'mode': mode, 'evidence': evidence}
