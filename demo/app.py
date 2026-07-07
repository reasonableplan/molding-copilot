"""demo/app.py — 現答 라이브 데모 (Streamlit · HF Spaces).

사전계산 파생값(assets/shots.json) 재생 데모. KAMP 원본 데이터와 로컬 LLM 은
라이선스·자원 제약으로 포함하지 않는다 — 판단·숫자는 전부 로컬 ML 파이프라인
(LOO 교차적합)의 사전계산값이고, 코파일럿 문장은 gemma3:4b 로 사전 생성해 그대로
보여준다(재생성 없음). 재현 절차: repo README '데이터 출처 · 재현'.
"""
import json
import os

import altair as alt
import pandas as pd
import streamlit as st

REPO = "https://github.com/reasonableplan/molding-copilot"
CITED_COLOR, REST_COLOR = "#2C5BE0", "#C9CDD4"
Z_CITE = 2.0  # src/grounding.py Z_CITE_GATED 와 동일 — 근거로 인용되는 |z| 하한

st.set_page_config(page_title="現答 — 측정된 신뢰 데모", layout="wide")


@st.cache_data
def load_shots():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "shots.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


data = load_shots()
shots = data["shots"]
trust = data["trust"]
defects = [s for s in shots if s["groundTruth"] != "정상"]
normals = [s for s in shots if s["groundTruth"] == "정상"]
detected = sum(s["isAnomaly"] for s in defects)
false_pos = sum(s["isAnomaly"] for s in normals)

st.title("現答 — 측정된 신뢰의 사출성형 결함진단")
st.markdown(
    "이상이면 **어느 공정변수가 몇 σ 벗어났는지** 근거를 대고, 근거가 부족하면 "
    "**\"모른다\"고 기권**한다. 그 정직함(오탐율 ≤ α)을 split-conformal 로 수치 보장한다. "
    f"[GitHub]({REPO})"
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("진짜결함 recall", f"{trust['recallSup']}%")
m1.caption(f"지도 게이트 · 비지도 {trust['recallUnsup']}% 대비 (eda/16)")
m2.metric("오탐율 보장", f"{trust['fprGuarantee']}% ≤ {trust['alpha']}%")
m2.caption("split-conformal · K=300 평균 (eda/06)")
m3.metric("confidently-wrong", f"{trust['cwVanilla']}% → {trust['cwOurs']}%")
m3.caption("vanilla LLM 대비, 같은 gemma3:4b (eda/09)")
m4.metric("이 데모 표본", f"{detected}/{len(defects)} 검출")
m4.caption(f"오탐 {false_pos}/{len(normals)} · 기권 {len(defects) - detected} (LOO out-of-sample)")

st.divider()

pick, result = st.columns([1, 3], gap="large")

with pick:
    st.subheader("Shot 선택")
    default_id = st.query_params.get("shot")
    target = next((s for s in shots if s["id"] == default_id), None)
    labels = ["가스", "미성형", "정상"]
    category = st.radio("실제 라벨", labels,
                        index=labels.index(target["groundTruth"]) if target else 0,
                        help="KAMP 검사 라벨 기준. 진단은 라벨을 보지 않는다.")
    pool = [s for s in shots if s["groundTruth"] == category]
    idx = next((i for i, s in enumerate(pool) if s["id"] == default_id), 0)
    shot = st.selectbox("shot", pool, index=idx,
                        format_func=lambda s: f"{s['id']} · conformal p={s['pValue']:.4f}")
    st.caption(
        "결함 shot 진단은 그 shot 을 학습에서 뺀 분류기로 재적합(LOO)한 "
        "out-of-sample 결과 — 보고 recall 과 같은 기준이다."
    )

with result:
    gt = shot["groundTruth"]
    if shot["isAnomaly"]:
        st.error(f"판정: 이상 — 추정 결함모드 [{shot['mode']}] · "
                 f"conformal p = {shot['pValue']:.4f} ≤ α = {shot['alpha']}")
        if gt == "정상":
            st.caption(f"실제 라벨: 정상 — α={trust['alpha']}% 한도 안에서 허용된 오탐 사례.")
        else:
            agree = "모드까지 일치" if shot["mode"] == gt else f"모드는 {shot['mode']}로 추정"
            st.caption(f"실제 라벨: {gt} — 검출 성공, {agree}.")
    else:
        st.info(f"판정: 근거불충분 — conformal p = {shot['pValue']:.4f} > α = {shot['alpha']}, "
                "단정하지 않고 기권")
        if gt == "정상":
            st.caption("실제 라벨: 정상 — 올바른 통과.")
        else:
            st.caption(f"실제 라벨: {gt} — 게이트 점수가 판정 임계에 못 미쳐 기권. "
                       "틀린 확신 대신 \"모른다\"가 이 시스템의 설계다.")

    ev_chart, ev_table = st.columns([5, 4], gap="medium")
    with ev_chart:
        zdf = pd.DataFrame(shot["zbars"])
        cited = {e["var"] for e in shot["evidence"]}
        zdf["cite"] = zdf["var"].map(lambda v: "인용 근거" if v in cited else f"|z| < {Z_CITE}")
        bars = alt.Chart(zdf).mark_bar().encode(
            x=alt.X("z:Q", title="z — 정상 대비 표준편차"),
            y=alt.Y("var:N", sort=None, title=None, axis=alt.Axis(labelLimit=230)),
            color=alt.Color("cite:N", title=None,
                            scale=alt.Scale(domain=["인용 근거", f"|z| < {Z_CITE}"],
                                            range=[CITED_COLOR, REST_COLOR])),
        ).properties(height=250)
        st.altair_chart(bars, use_container_width=True)

    with ev_table:
        if shot["evidence"]:
            edf = pd.DataFrame(shot["evidence"])[["var", "observed", "normalMean", "normalSd", "z"]]
            edf.columns = ["변수", "관측", "정상 μ", "정상 σ", "z"]
            st.dataframe(edf, hide_index=True, use_container_width=True)
        elif not shot["isAnomaly"]:
            st.caption("인용된 근거 없음 — 게이트가 기권하면 원인을 서술하지 않는다.")
        else:
            st.caption(f"인용 가능한 근거 없음 — |z| ≥ {Z_CITE} 인 공정변수가 없다.")

    if shot["prescription"]:
        st.markdown(shot["prescription"])

    st.markdown(f"> {shot['copilot']}")
    st.caption("코파일럿 문장: gemma3:4b 사전 생성 — EVIDENCE 수치만 인용 가능, 판단·숫자는 "
               "전부 ML (grounding-by-construction). 기권 시 LLM 을 거치지 않는다.")

st.divider()
with st.expander("이 데모가 만들어진 방식 — 무엇이 실측이고 무엇이 아닌가"):
    st.markdown(f"""
- **데이터**: KAMP 사출성형(실제 현대 CN7 부품, 650톤 우진2호기). 원본 데이터는 재배포
  제한으로 이 데모에 포함하지 않는다 — 위 결과는 로컬에서 사전계산한 **파생값 재생**이다.
  전체 재현 절차는 [repo README]({REPO}) 참고.
- **누설 차단**: 결함 shot 진단은 그 shot 을 학습에서 제외한 분류기로 재적합(LOO 교차적합,
  eda/18). 정상 shot 은 학습에 쓰지 않은 held-out 20% 에서만 뽑았다.
- **역할 분리**: 이상/기권 판정과 모든 수치는 결정론적 ML(지도 GBM 게이트 + split-conformal +
  z-편차)이 낸다. LLM(gemma3:4b)은 계산된 EVIDENCE 를 한국어로 표현만 한다 — 숫자를 지어낼
  통로가 없다.
- **헤드라인 지표 출처**: recall 58→89% (eda/16), 오탐율 4.94%≤5% (eda/06, K=300),
  confidently-wrong 67.5→7.5% (eda/09). 전부 repo `outputs/` 에 산출 로그가 있다.
""")
