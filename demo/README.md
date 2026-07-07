---
title: Hyundab Molding Copilot Demo
emoji: 🏭
colorFrom: gray
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# 現答 (현답) — 측정된 신뢰 데모

사출성형 결함진단 코파일럿의 라이브 데모. 이상이면 공정변수 근거(z-편차)를 대고,
근거가 부족하면 "모른다"고 기권하며, 오탐율 ≤ α 를 split-conformal 로 수치 보장한다.

- 소스·재현 절차: https://github.com/reasonableplan/molding-copilot
- KAMP 원본 데이터는 재배포 제한으로 포함하지 않는다 — 이 데모는 로컬에서 사전계산한
  파생값(`assets/shots.json`)을 재생한다. 결함 진단은 LOO 교차적합 out-of-sample 기준.
- 코파일럿 문장은 로컬 gemma3:4b 로 사전 생성 — 판단·숫자는 전부 결정론적 ML.

## 로컬 실행

```
pip install -r requirements.txt
streamlit run app.py
```
