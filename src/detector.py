"""detector.py — 결정론적 진단 코어 (LLM 없이 동작).
[게이트] 이상/모른다 판정 + [변수 근거] 어느 공정변수가 정상서 벗어났나.
게이트 score는 두 모드:
  - 비지도(기본): 정상으로부터의 Mahalanobis 거리 (라벨 불필요).
  - 지도(X_defects 주면): 정상+결함 분류기의 불량확률 (recall↑, eda/16: 58→89%).
어느 모드든 conformal 보장(오탐율<=α)은 동일 — calib/test 정상이 학습에 안 들어가므로(보장은 score 무관).
설명(z-편차 근거 + 모드추정)은 게이트 모드와 무관하게 정상분포 기준으로 동일.
copilot은 이 diagnose() 출력을 자연어로 표현만 한다 (숫자 생성 X = hallucination 차단).
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from src.conformal import ConformalGate
from src.grounding import Z_CITE_GATED, infer_mode


class Detector:
    def __init__(self, alpha=0.05, top_k=3, z_threshold=Z_CITE_GATED):
        self.alpha, self.k, self.zt = alpha, top_k, z_threshold

    def fit(self, X_fit: pd.DataFrame, X_calib: pd.DataFrame, X_defects: pd.DataFrame = None):
        """X_fit: 정상(모델 적합), X_calib: 정상(conformal 보정). 분리 필수.
        X_defects: 결함 라벨(있으면 지도 게이트 — recall↑). None이면 비지도 Mahalanobis."""
        self.cols = list(X_fit.columns)
        self.mu = X_fit.mean(); self.sd = X_fit.std().replace(0, np.nan)
        if X_defects is not None and len(X_defects):
            self.clf = make_pipeline(StandardScaler(),
                                     GradientBoostingClassifier(random_state=0))
            self.clf.fit(pd.concat([X_fit[self.cols], X_defects[self.cols]]),
                         np.r_[np.zeros(len(X_fit)), np.ones(len(X_defects))])
            self.gate_mode = 'supervised'
        else:
            self.clf = None
            self.scaler = StandardScaler().fit(X_fit)
            self.cov = LedoitWolf().fit(self.scaler.transform(X_fit))
            self.gate_mode = 'mahalanobis'
        self.gate = ConformalGate(self.alpha).fit(self._score(X_calib))
        return self

    def _score(self, X):
        if self.clf is not None:                                  # 지도: 불량확률(높을수록 이상)
            return self.clf.predict_proba(X[self.cols])[:, 1]
        return self.cov.mahalanobis(self.scaler.transform(X[self.cols]))   # 비지도: 제곱 거리

    def diagnose(self, shot: pd.Series) -> dict:
        s = float(self._score(shot[self.cols].to_frame().T))
        p = float(self.gate.p_value(s))
        z = ((shot[self.cols] - self.mu) / self.sd).dropna()
        ranked = z.reindex(z.abs().sort_values(ascending=False).index)

        if p > self.alpha:                          # ── conformal 게이트: 모른다
            return {'verdict': '근거불충분', 'is_anomaly': False, 'p_value': p,
                    'alpha': self.alpha, 'score': s, 'gate_mode': self.gate_mode,
                    'mode': None, 'evidence': []}

        cited = ranked[ranked.abs() >= self.zt].head(self.k)
        evidence = [{'var': v, 'observed': round(float(shot[v]), 2),
                     'normal_mean': round(float(self.mu[v]), 2),
                     'normal_sd': round(float(self.sd[v]), 2), 'z': round(float(z[v]), 2)}
                    for v in cited.index]
        mode = infer_mode(evidence)      # |z|가중 투표(grounding과 공유)
        return {'verdict': '이상', 'is_anomaly': True, 'p_value': p, 'alpha': self.alpha,
                'score': s, 'gate_mode': self.gate_mode, 'mode': mode, 'evidence': evidence}
