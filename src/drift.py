"""drift.py — conformal test martingale(Simple Jumper) 드리프트 조기경보.

검출 칸(conformal.py)의 split-conformal p-value 스트림을 받아, 교환가능성(IID) 검정
마틴게일을 누적한다. 정상 스트림에서는 자본 S_n이 1 근처에 머물고, 분포가 드리프트하면
S_n이 발산한다 → 임계 c를 넘으면 '드리프트 경보'.

핵심 보장 — Ville 부등식: 교환가능성 하에서 P(sup_n S_n >= c) <= 1/c.
즉 임계 c=100 이면 오경보 확률 <= 1% 가 분포가정 없이 유한표본 보장된다.
(검출 칸의 오탐율<=alpha 보장이 '예측' 칸으로 한 칸 올라간 것 — 같은 measured-trust 척추.)

Simple Jumper (Vovk 2021, "Retrain or not retrain"): 자본 3계좌 C_{-1},C_0,C_{+1}.
betting f_eps(p)=1+eps(p-0.5), eps in {-1,0,+1}. eps=-1은 p가 작은 쪽(이상↑)에, +1은
큰 쪽에 건다. jumping rate J로 매 스텝 재분배 → 드리프트 시점에 적응.
"""
import numpy as np

EPS = np.array([-1.0, 0.0, 1.0])


class SimpleJumper:
    def __init__(self, J=0.01):
        self.J = J            # jumping rate (Vovk 권장 0.01)
        self.reset()

    def reset(self):
        self.C = np.full(3, 1 / 3)   # C_{-1}, C_0, C_{+1}
        self.S = 1.0                 # 마틴게일 값 = 총자본 (S_0 = 1)
        return self

    def update(self, p):
        """conformal p-value 하나를 받아 자본 갱신, 갱신된 S 반환."""
        total = self.C.sum()
        self.C = (1 - self.J) * self.C + (self.J / 3) * total   # ① 재분배(jump)
        self.C = self.C * (1 + EPS * (p - 0.5))                 # ② 베팅 f_eps(p)
        self.S = self.C.sum()                                   # ③ 총자본
        return self.S

    def run(self, pvals, threshold=None, reset_on_alarm=False):
        """p-value 스트림 전체를 흘려 S_n 궤적 반환.
        threshold 주면 첫 교차 인덱스 alarms 에 기록. reset_on_alarm=True면 경보 후 재시작.
        cap: 발산 시 오버플로 방지(임계보다 훨씬 큰 1e12에서 절단 — 이미 경보 상태)."""
        traj = np.empty(len(pvals))
        alarms = []
        for i, p in enumerate(pvals):
            s = self.update(p)
            if s > 1e12:                       # 오버플로 가드
                s = 1e12
                self.C *= 1e12 / self.C.sum()
            traj[i] = s
            if threshold is not None and s >= threshold:
                alarms.append(i)
                if reset_on_alarm:
                    self.reset()
        return traj, alarms
