"""copilot.py — 자연어 표현층. detector.diagnose() 결과를 한국어로 표현한다.
LLM은 EVIDENCE 안의 수치만 사용 (숫자 생성 X). 게이트면 LLM 없이 결정론적 '모른다'.
이게 grounding-by-construction: 모델이 데이터를 지어낼 통로가 없다.
"""
import json
import ollama

SYS = (
    "너는 사출성형 품질 진단 보조다. 아래 EVIDENCE(정상 대비 벗어난 공정변수)만 근거로 "
    "한국어 2~3문장으로 왜 불량인지 설명한다. 규칙: (1) EVIDENCE에 없는 변수·수치·원인을 "
    "절대 지어내지 마라. (2) 모든 수치는 EVIDENCE에서 그대로 인용하라. (3) mode가 있으면 "
    "그 결함유형('가스'=과열, '미성형'=충전부족)을 언급하라. (4) 추측·일반론 금지. "
    "(5) 반드시 순수 한국어로만 작성하라. 한자·중국어·일본어 절대 금지(영문 변수명은 허용)."
)


class Copilot:
    def __init__(self, detector, model="gemma3:4b"):
        self.det = detector
        self.model = model

    def ask(self, shot, question="이 제품이 왜 불량인가?") -> dict:
        d = self.det.diagnose(shot)
        if not d['is_anomaly']:               # ── conformal 게이트: LLM 거치지 않음
            return {'answer': "근거 불충분: 측정 공정변수상 정상과 구분되지 않습니다. "
                              "원인을 단정할 수 없습니다 (지어내지 않음).",
                    'grounded': True, 'diagnosis': d}
        ev = {'mode': d['mode'], 'evidence': d['evidence']}
        msg = [{'role': 'system', 'content': SYS},
               {'role': 'user', 'content': f"질문: {question}\nEVIDENCE:\n{json.dumps(ev, ensure_ascii=False)}"}]
        out = ollama.chat(model=self.model, messages=msg, options={'temperature': 0})
        return {'answer': out['message']['content'].strip(), 'grounded': True, 'diagnosis': d}
