"""conformal.py — split(inductive) conformal anomaly gate.

정상 calibration 점수만으로 임의의 anomaly score를 보정된 p-value로 바꾼다.
p(x) = (1 + #{calib_i >= score_x}) / (n_calib + 1)   (높을수록 이상인 score 기준)
교환가능성 하에서 P(정상을 이상으로 오판) <= alpha 가 유한표본 보장된다.
→ |z|>3 같은 임의 임계 대신, 오탐율을 우리가 정하고 그게 지켜진다.
"""
import numpy as np


class ConformalGate:
    def __init__(self, alpha=0.05):
        self.alpha = alpha            # 허용 오탐율(정상을 이상이라 할 확률 상한)

    def fit(self, calib_scores):
        """calib_scores: 정상 calibration shot들의 anomaly score (높을수록 이상)."""
        self.calib = np.sort(np.asarray(calib_scores, float))
        self.n = len(self.calib)
        return self

    def p_value(self, score):
        s = np.atleast_1d(np.asarray(score, float))
        # #{calib >= s} = n - (calib < s 개수)
        ge = self.n - np.searchsorted(self.calib, s, side='left')
        p = (1 + ge) / (self.n + 1)
        return p if p.size > 1 else float(p[0])

    def is_anomaly(self, score):
        p = self.p_value(score)
        return (np.asarray(p) <= self.alpha)
