"""causal.py — robust 인과 주석 (LiNGAM 결과 + 도메인 판단의 교집합만 신뢰).
DirectLiNGAM(eda/11)을 통째로 믿지 않는다. LiNGAM과 도메인이 *일치*하면 신뢰,
LiNGAM이 도메인과 충돌하면(아티팩트) 도메인으로 override. 각 주석에 basis(근거)를 남긴다.
용도: 진단된 원인변수가 'lever(직접 조정 가능)'인지 'symptom(상류가 원인)'인지 구분해
처방을 바로잡는다 — 증상을 조정하라 하지 않고 상류 lever를 점검하라 한다.
"""

# 진단 시그니처 변수에 한정한 큐레이션 (전체 그래프 X — 신뢰 가능한 것만)
CAUSAL_ROLE = {
    'Mold_Temperature_4': {
        'role': 'symptom',
        'upstream': ['Clamp_Open_Position', 'Plasticizing_Position', 'Max_Injection_Pressure'],
        'basis': 'LiNGAM+도메인 일치 (금형온도는 클램프/가소화/사출압의 결과)'},
    'Mold_Temperature_3': {
        'role': 'symptom',
        'upstream': ['Mold_Temperature_4', 'Clamp_Open_Position'],
        'basis': 'LiNGAM+도메인 일치 (Temp_4가 Temp_3을 야기)'},
    'Max_Injection_Speed': {
        'role': 'lever',
        'upstream': [],
        'basis': '도메인 override (LiNGAM은 최하류=아티팩트; 사출속도는 작업자 setpoint)'},
}


def annotate(var: str) -> dict:
    """미확인 변수는 보수적으로 lever 취급(증상으로 과잉 단정하지 않음)."""
    return CAUSAL_ROLE.get(var, {'role': 'lever', 'upstream': [],
                                 'basis': '기본 (인과 미확인 → 직접 lever로 보수적 취급)'})
