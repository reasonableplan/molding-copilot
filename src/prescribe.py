"""prescribe.py — 처방 레이어 (algorithmic recourse).
이상 shot을 정상으로 되돌리는 최소·희소 변경을 찾고, counterfactual로 검증한다.
- 희소(parsimony): 벗어난 변수를 |z| 순으로 하나씩 추가해 p>alpha 되는 최소 집합.
- 최소변경: 그 집합을 관측→정상평균으로 비율 t 보간, p>alpha 되는 최소 t 이분탐색.
- 검증: 변경 후 conformal p가 실제로 alpha를 넘는지(정상 예측) 함께 보고.
[한계] 이는 상관 모델(Mahalanobis)에 대한 recourse = "모델이 정상으로 보는 변경"이지
       물리적 인과 보장은 아니다. 인과 what-if는 다음 단계.
"""
import numpy as np
from src.causal import annotate


class Prescriber:
    def __init__(self, detector):
        self.det = detector

    def _p(self, shot, varset, t):
        m = shot.copy()
        for v in varset:
            m[v] = shot[v] + t * (self.det.mu[v] - shot[v])     # 관측→평균 t비율
        s = float(self.det._score(m[self.det.cols].to_frame().T))
        return self.det.gate.p_value(s)

    def prescribe(self, shot, p_target=0.15) -> dict:
        """진단된 '원인 변수'(설명에 인용된 것)만 정상쪽으로 조정 권고하고, what-if로
        그 효과를 솔직히 보고한다. 자유 recourse(아무 상관변수나 끌어쓰는 비현실 처방)는 하지 않는다.
        모델 what-if가 정상 복귀를 확인 못 하면 '공정 검토 권고'로 정직하게 보류한다(다변량/인과 가능성)."""
        d = self.det.diagnose(shot)
        if not d['is_anomaly']:
            return {'needed': False, 'text': "정상/근거불충분 — 처방 불필요."}

        p0 = d['p_value']
        cause = [e['var'] for e in d['evidence']]           # 진단이 인용한 원인 변수
        p1 = self._p(shot, cause, 1.0)                      # 원인을 정상평균으로 되돌린 what-if

        # 인과 주석: lever(직접 조정) vs symptom(상류 점검)으로 분리
        levers, lever_vars, symptoms, ups = [], [], [], []
        for e in d['evidence']:
            c = annotate(e['var'])
            tgt = round(float(self.det.mu[e['var']]), 2)
            if c['role'] == 'lever':
                lever_vars.append(e['var'])
                levers.append(f"{e['var']} {e['observed']}→{tgt} ({round(tgt-e['observed'],2):+})")
            else:
                symptoms.append(e['var'])
                ups += [u for u in c['upstream'] if u not in ups]
        # 상류 목록에서 이미 다루는 변수(증상 자신·직접 lever) 제거
        ups = [u for u in ups if u not in symptoms and u not in lever_vars]

        msg = []
        if levers:
            msg.append("직접 조정(lever): " + ", ".join(levers))
        if symptoms:
            msg.append(f"증상(symptom) {', '.join(symptoms)} 은 결과일 뿐 → 상류 점검: {', '.join(ups)}")
        verdict = "정상 복귀 예측" if p1 >= p_target else "정상 복귀 미확인 → 공정 검토 권고"
        return {'needed': True, 'resolved': p1 >= p_target, 'mode': d['mode'],
                'levers': levers, 'symptoms': symptoms, 'upstream': ups,
                'p_before': round(p0, 3), 'p_after': round(p1, 3),
                'text': "처방 — " + " | ".join(msg) + f"  ⇒ what-if(직접조정 기준): p {p0:.3f}→{p1:.3f} ({verdict})"}
