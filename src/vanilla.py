"""vanilla.py — 베이스라인: grounding 엔진 없이 LLM 단독 진단.
같은 shot의 raw 공정변수 값만 주고 '불량인지/어느 변수가 이상인지/정상평균/결함모드'를 구조화 출력시킨다.
정상 분포 통계가 없으므로 '정상평균'을 추측(=지어냄)할 수밖에 없다 → grounding 부재를 정량화하는 대조군.
"""
import json
from typing import Optional
import ollama
from pydantic import BaseModel


class Diag(BaseModel):
    is_defect: bool
    top_variable: Optional[str] = None
    observed_value: Optional[float] = None
    claimed_normal_mean: Optional[float] = None   # 엔진 없이는 알 수 없음 → 추측
    defect_mode: Optional[str] = None              # "가스" | "미성형" | null


SYS = (
    "너는 사출성형 품질 진단 AI다. 아래 한 제품(shot)의 공정변수 측정값만 보고 "
    "불량 여부와 원인을 진단하라. 불량이면 가장 이상한 변수(top_variable), 그 관측값, "
    "그 변수의 정상 평균(claimed_normal_mean), 결함모드(가스=과열 / 미성형=충전부족)를 채워라. "
    "정상으로 보이면 is_defect=false. 반드시 JSON 스키마로만 답하라."
)


class VanillaDiagnoser:
    def __init__(self, model="gemma3:4b"):
        self.model = model

    def diagnose(self, shot, cols) -> dict:
        vals = "\n".join(f"{c} = {float(shot[c]):.2f}" for c in cols)
        msg = [{'role': 'system', 'content': SYS},
               {'role': 'user', 'content': f"공정변수 측정값:\n{vals}\n\n진단(JSON):"}]
        out = ollama.chat(model=self.model, messages=msg,
                          format=Diag.model_json_schema(), options={'temperature': 0})
        try:
            return Diag.model_validate_json(out['message']['content']).model_dump()
        except Exception:
            return Diag(is_defect=False).model_dump()
