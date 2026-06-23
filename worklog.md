# worklog — molding-copilot

진실의 원천. 시간 역순 아님(위→아래 진행순).

---

## 2026-06-23 — 착수, 0~1단계

### 방향 확정
- 여러 방향(eval 하네스/memory 엔진/grounded/model-level/AX) 탐색 끝에 **제조 AX**로 확정.
- KAMP 데이터는 무료로 열려 있음 확인. 사출성형 데이터 4종 다운로드.
- 4종 판정: **Molding(불량+Reason)=척추**, 예지보전(라벨없음)=기법참조, 콘택트렌즈=Phase2, 공급망=버림.
  - 처음엔 크기만 보고 예지보전을 척추로 골랐다가, 실데이터 확인 후 **라벨+Reason 있는 Molding으로 변경**.

### 0단계 — 데이터 확인 (완료)
- 환경: Python 3.13.7 / pandas 2.3.3 / sklearn 1.7.2 [verified].
- zip 4개는 `00/` 루트. Molding zip 해제 → `data/Dataset_Molding/dataset/` (CSV 8종).
- 가이드북 PDF/문서: **4개 zip 어디에도 없음.** 메타데이터가 대체.
- `labeled_data.csv` 인코딩 = **utf-8** (cp949로 읽으면 0xec 에러). 콘솔 한글 깨짐은 표시 문제일 뿐.
- 라벨 분포 [verified]:
  - 전체 7,996행 / 불량(N) 71 / Reason: 가스 35·초기허용불량 20·미성형 16.
  - **CN7 부품군 6,736행 / 불량 39 / Reason: 초기허용불량 20·가스 13·미성형 6.**
- 피처: 표준화된 24 공정변수(moldset/supervised 셋), 금형온도 5~12·Switch_Over 등 상수 칼럼 제거됨.
- 파일 역할: unlabeled_cn7(35,239 학습) / supervised_label_cn7(6,736·39불량 검증) / labeled_data(사유).

### 1단계 — 타당성 게이트 (완료)  → `outputs/02_separability.txt`
- 정상 분포로 표준화 후 결함군 평균 z-편차(효과크기) 측정.
- **초기허용불량 20**: z=+17(Filling_Time) 등 압도적 분리 = cold-start 확정 → **타깃 제외.**
- **가스 13**: Mold_Temp_4 z=+3.1, Mold_Temp_3 z=+2.6 → 금형온도↑ 상관, 도메인 부합. 약하지만 잡힘.
- **미성형 6**: Max_Injection_Speed z=+4.1 → 사출속도 이상. 단 **1건 max|z|=0.8 = 안 보임.**
- 오탐 하한: 정상의 **4.9%**가 어떤 변수든 |z|>3.
- **결론: 가능, 단 검출 천장 존재.** 진짜 타깃 = 가스+미성형 19개(0.28%). 정직한 eval로 천장 보고가 컨셉.

### 데이터 환경 조사 + 선행연구 (2026-06-23)
- zip 정리: Molding(추출완료)·공급망(버림) 삭제. 예지보전·콘택트렌즈(Phase2) zip만 00/ 루트에 잔존.
- **우리 데이터 = 발표된 학술 벤치마크** (windshield side molding, KAIST/UNIST/EPM 2020):
  - [Oxford JCDE 2023]: 지도학습+클래스밸런싱손실(SL-CBL) → **F1 97.3%, AUC 0.99** (정상3956/불량28, 141:1).
  - [Wiley 2024]: 오토인코더+ML. [MDPI 2025]: XAI. [Springer 2025]: 불균형+fuzzy boundary(=우리 "미성형 안보임" 천장).
- **[정정]** 0/1단계서 "양성 적어 지도분류 불가"는 과했음 → 지도+focal/F1손실로 F1 97% 가능(단 recall 분산 큼).
- **[전략 확정]** 모델링은 이미 풀려 commodity. 2주를 모델에 쓰면 논문 재현일 뿐. → **벤치마크 수준만 빠르게
  재현해 기준점 삼고, 노력의 대부분을 레이어 2(grounded 설명)·3(정직한 eval)에 투입.** 논문엔 그게 없음.
- 추가 데이터: airtlab iGuzzini(렌즈 사출 1451) = Phase2 일반화검증 선택. AI Hub/공공데이터포털 = MVP 밖.
  **결론: 지금 데이터 더 받지 않음.** 단일공정 depth 유지.

### 1.5단계 — 시각화 (완료) → outputs/figures/{hist,scatter,pca}.png
- 산점도: 가스=금형온도27.5 위로 분리, 미성형=대부분 우상단+1개 정상한복판, 초기허용=좌하단 저온 cold-start.
- PCA: 가스 PC2축↑ / 초기허용 PC1축 우하단(서로 다른 방향) / 미성형1개 정상띠에 묻힘 / 정상도 좌측 outlier꼬리.
- 단일변수로 전부 가르는 것 없음 → 다변량 + "모른다" 게이트가 실제 맹점으로 정당화.

### 2단계 — 베이스라인 (완료) → outputs/04_baseline.txt, figures/anomaly_scores.png
- (A) PCA-AE(k=10, 분산99%) 재구성오차 anomaly score. 정상70%로 fit.
  - **진짜(가스+미성형 19): PR-AUC 0.236 / ROC 0.769 / recall@FPR5% 0.42**
  - 쉬운(초기허용 20): ROC 0.999 / 전체39: ROC 0.887  ← 섞으면 부풀려짐(=confidently-wrong 데모)
- (B) 지도 LR(balanced) 5-fold 전체39: PR-AUC 0.693 / **ROC 0.985** (발표 SL-CBL F1~0.97/AUC~0.99 궤도 일치).
- **[핵심 발견]** AE가 가스를 못 잡음(빨강이 정상 봉우리에 묻힘): k=10이 가스변동 방향(금형온도)까지 부분공간에
  흡수 → AE는 부분공간 밖만 이상으로 봄. 지도학습은 "가스다"를 알려주면 잡음(ROC0.985). → **검출=지도,
  AE=grounding/해석 역할 분리.** 맹점은 약함(미성형 0/6이 정상중앙값 이하 = 다변량이 단변량 맹점 보완).

### 레이어2 — grounded 설명 (완료) → src/grounding.py, eda/05_grounding_eval.py, outputs/05_grounding.txt
- 엔진(GroundedExplainer): 정상 분포 fit → shot별 변수 z-편차 인용, |z|<3이면 '근거불충분(모른다)' 게이트,
  인용변수→도메인 시그니처(가스=과열/미성형=충전부족) 다수결로 결함모드 추정.
- **결과(진짜 19): 발화15/게이트4, grounded 적중 13/15=87%** (가스9/11·미성형4/4). naive(항상 다수결)=68%·게이트無.
  정상 오발화 5.0%(101/2010, |z|>3). 샘플: "Mold_Temperature_4=27.7(정상23.5±1.4,+3.1σ)→가스 일치" 등.
- 가치: 87>68 + 4건 정직한 기권(confidently-wrong 회피) + 오발화 5% 정직 보고 + 관측값 인용(non-hallucinated).
- **[정직한 한계]** 시그니처가 데이터 분리도서 도출 → 약한 순환성. 단 'shot별 어느 변수가 벗어났나'는 실측(순환 아님).
  개선: 외부 도메인 레퍼런스로 시그니처 독립 검증.

### 방향·기술 확정 (2026-06-23)
- **AX 서사 전면화 + 제조 플로우 스위트 비전.** 4개 KAMP 데이터 = 플로우 4단계: ①계획(공급망)
  ②셋업(콘택트렌즈 금형) ③가동(예지보전) ④검사(Molding, 현재). [조사: 한국 제조AX 업계 표준 패키징과 일치,
  코오롱베니트도 "품질검사부터 순차확대" = 우리 순서]
- **[경고/조사결과]** "통합 플랫폼"은 MS Copilot for Mfg/Bosch/Siemens/위즈코어 등 레드오션 → 솔로가 표방하면
  deck-ware. 화해: 통일 축을 "통합"이 아니라 **"measured trust(측정된 신뢰)"**로. 거대기업 누구도 "AI가
  거짓말하는지 측정"을 안 함 = 빈 자리. **빌드는 depth-first: ④ 끝까지 깊게 → 신뢰척추 추출 → ①②③ 로드맵/경량PoC.**
- **기술 스택 확정 (조사 후 채택):**
  - 🟢 Conformal Prediction → "모른다" 게이트: |z|>3 임계를 오탐율 통계보장으로 격상(분포가정無). [MAPIE/직접구현]
  - 🟢 RAGAS faithfulness/groundedness + LLM-as-judge → 레이어3 측정 표준화(답을 atomic claim으로 근거대조).
  - 🟢 Tool-calling grounding → copilot 배선: LLM은 NL front-end, grounding.py를 도구호출해 실측만 말함.
  - 🟡 SHAP/ShaTS(시계열 Shapley) → 레이어2 attribution 업그레이드(시간되면). 🔵 Counterfactual recourse → 스위트② 처방.
  - ⚠️ "Reasoning Trap"(arXiv 2510.22977): 추론강한 LLM이 tool hallucination 더함 → copilot 추론 좁게 제약.
  - 선행연구 차별: EIAD=멀티모달 vision 이상설명. 우리=tabular+정직성 측정.

### 레이어3-a — conformal "모른다" 게이트 (완료) → src/conformal.py, eda/06, outputs/06_conformal.txt
- Mahalanobis 거리(LedoitWolf, 정상 fit) = anomaly score → split-conformal p-value → α 게이트.
- 검증(K=300 split 평균): α=0.05→오탐율 4.94%≤5% OK, α=0.01→0.92%≤1% OK. 보장은 marginal(단일split는 95%분위 6.3%까지 흔들림=정상 표본오차, 정직 표기).
- 진짜결함 recall@α0.05 = **58%** (step2 PCA-AE 42% 대비↑, 가스 금형온도 거리기여로 잡힘). 모드별 가스54%/미성형67%/cold100%.
- 핵심: 오탐율이 '관측→보장'. 정직한 천장(58%, 42% 누락)을 명시적·보장된 trade-off로 드러냄.
- 다음 통합: copilot이 [conformal 게이트→이상이면 grounding 설명]으로 조립 (grounding의 |z|>3 게이트 대체).

### 레이어3-b — grounded copilot (완료) → src/detector.py, src/copilot.py, eda/07~08
- Detector: [conformal 게이트(Mahalanobis+LedoitWolf)] + [변수 z-근거 + |z|가중 모드추정]. diagnose()가 구조화 evidence 반환.
  - 검증(07): 진짜19→발화11/게이트8, 모드적중 9/11=82%. 정상은 p=0.84~0.98로 명확 기권. [버그수정: 모드투표를
    단순다수결→|z|가중으로, 미성형이 약한 금형온도 2개에 밀려 '가스' 오판되던 것 해결]
- Copilot: detector evidence를 LLM이 **그 수치 안에서만** 자연어 표현(grounding-by-construction), 게이트면 LLM 거치지 않고 결정론적 '모른다'.
- **[모델 A/B 발견]** 같은 evidence라도 모델별 실패: qwen2.5:7b=중국어누출, llama3.1:8b=숫자오기(6.45→5.45),
  qwen3:8b=인과오귀속(최강신호 무시), **gemma3:4b=충실+순수한국어 → 채택.** = grounding 줘도 LLM은 샌다(=3c가 측정할 것).

### 레이어3-c — faithfulness 대조 (완료) → src/vanilla.py, eda/09, outputs/09_faithfulness.txt
- 방법: RAGAS faithfulness 착안하되 로컬judge 약함 → ground truth(detector) 대고 결정론적 검증. 같은 모델
  gemma3:4b를 엔진 有(우리)/無(vanilla=raw수치만 구조화진단)로만 가름.
- **헤드라인 [verified]:**
  - 진짜결함19 모드: 우리=정답9/기권8/오답2  vs  vanilla=정답2/기권0/오답17.
  - 정상범위 날조: 우리 0%(실측) vs vanilla 75%(3/4, n작음 caveat).
  - 정상40건 confidently-wrong: **우리 8%(α5% 보장) vs vanilla 68%(27/40).**
- 해석: vanilla는 정상의 68%를 불량 단정+진짜결함 17/19 오진(정상baseline 모름). 우리는 모르면 기권.
- caveat: gemma3:4b가 구조화출력서 top_variable 자주 None punt(=엔진없이 baseline 못만듦의 증거). robust지표=모드·CW.

### ★ 레이어3 완성 = 3-레이어 MVP 코어 닫힘
- 보장된 검출(conformal) + grounded 설명(detector+copilot) + 측정된 정직성(vanilla 대조). 핵심 서사 완결.

### AX 가치사다리 확장 (서칭) — 진단 다음 칸들
- 성숙도: 서술→[진단=현재]→예측→처방→자율(agentic). 사용자 "진단서 끝나면 안됨" 지적 타당.
- 핵심: 사다리 오를수록 confidently-wrong 비용 폭발(진단=1분/처방=스크랩/자율=설비사고) → 우리 conformal
  보장게이트가 곧 "bounded autonomy/사람 에스컬레이션" = 자율로 가는 신뢰 substrate. 세로축이 가로(스위트)보다 강함.
- [경고] 우리 grounding은 상관(correlation)이지 인과 아님 → 신뢰가능 처방엔 causal/what-if 필요(arXiv 2505.01445).

### 처방 레이어 (완료, 정직한 결론) → src/prescribe.py, eda/10, outputs/10_prescribe.txt
- algorithmic recourse: 진단된 원인변수만 정상평균으로 되돌리는 방향 권고 + what-if(conformal p 재계산)로 효과 검증.
- [반복 정정] (1)최소-t 이분탐색→경계 degenerate 마이크로조정(−0.01). (2)자유 recourse→비현실(ScrewRPM −170 등
  아무 상관변수나 끌어씀). → 최종: 진단 원인변수만, 정직한 what-if 보고.
- **[핵심 발견] 진단 원인을 정상화해도 what-if p 거의 안 오름**(미성형 0.001→0.006, 가스 0.014→0.025). 결함이
  다변량이라 인용 3변수 고쳐도 24차원 Mahalanobis 안 줄어듦. → 발화11 전부 방향권고 O, 정상복귀 확인 0,
  **전부 '공정검토 권고'**(오버프로미스 안 함). = **상관 recourse 한계 실증 = 인과 모델이 검증된 다음 칸.**
- 가치: 처방도 "정직" 브랜드 유지 — 방향은 근거있게, 안 통하면 솔직히 공정검토. 자동수리 사칭 안 함.

### 인과 레이어 (탐색 완료, 정직한 천장) → eda/11_causal.py, outputs/11_causal.txt  [lingam 1.12.2 설치]
- DirectLiNGAM(정상 3348shot, 표준화) → 공정변수 인과순서 + adjacency.
- ✅ **값진 발견: 금형온도=증상.** Mold_Temp_4(순서9)는 Clamp_Open/Plasticizing_Position·사출압에 의해 야기,
  Mold_Temp_3(순서16)은 Mold_Temp_4가 일으킴. → "가스 원인=금형온도"는 하류 symptom = **처방이 왜 안통했는지 설명**.
- 🚩 **빨간불: Max_Injection_Speed가 순서23/23(최하류), Injection_Time이 일으킨다고 나옴 = 아티팩트.** 속도는
  setpoint인데 시간에 의해 야기? 속도×시간≈부피 기계적결합을 인과로 오인. 단일라인 변동적음+정의적상관이 가정 위반.
- **정직한 결론: 관측 단일라인 데이터의 인과탐색은 천장. 일부구조(증상)는 건지나 결합된 setpoint엔 엉터리방향.**
  신뢰 lever엔 도메인 prior(setpoint 라벨, 단 머신매뉴얼 없어 모호) 또는 개입(실험) 데이터 필요 = 솔로 관측 한계.
- 교훈: 진단→처방→인과로 오를수록 필요 가정·데이터↑, 관측데이터는 인과서 한계. (자율 함부로 약속 안 하는 정직함)

### 인과-인지 처방 (완료, A) → src/causal.py + prescribe.py 통합
- src/causal.py: LiNGAM∩도메인 교집합만 신뢰하는 큐레이션 주석. 금형온도=symptom(상류 Clamp_Open/Plasticizing/사출압),
  사출속도=lever(LiNGAM 아티팩트 도메인 override), 미확인=보수적 lever. 각 항목에 basis 명시.
- prescribe 통합: 인용 원인을 lever(직접 조정)/symptom(상류 점검)으로 분리. **가스 처방이 "금형온도 낮춰라"(헛다리)
  → "금형온도는 증상, 상류 Clamp_Open/Plasticizing/사출압 점검"으로 교정.** 미성형은 사출속도(lever) 직접+금형온도 상류.
- = 인과의 robust 부분만 제품화. 아티팩트(사출속도) 안 믿고 도메인 override. 정직 유지.

### ECOD(PyOD) vs Mahalanobis 비교 (완료, negative result) → eda/12, outputs/12_ecod_compare.txt [pyod 설치]
- 같은 conformal 게이트(α=0.05). Mahalanobis recall 58%(가스54%) vs ECOD 53%(가스46%). 둘 다 cold-start 100%, 미성형 67%.
- **차이는 noise(19개중 1개차, ±11%p) — 사실상 동률, Mahalanobis 약간 앞.** 이유: ECOD는 변수독립가정, 우리 결함은
  상관된 다변량(금형온도 동반)이라 공분산 쓰는 Mahalanobis가 유리. → **Mahalanobis 유지(측정해서 ECOD 떨굼).**
- ECOD 해석성은 진짜: 가스 top변수=Mold_Temp_3/4 + Plasticizing_Position(LiNGAM이 짚은 상류!). grounding 보조 가능하나 z-편차로 충분.
- conformal이 검출기 무관 재확인(ECOD도 그냥 꽂힘).

### ■ 마일스톤 정지 (D 선택, 2026-06-23) — 가치사다리 진단→처방→인과(robust) 통합 완료
- src/ 7모듈 248줄(conformal·detector·grounding·copilot·vanilla·prescribe·causal), eda/01~12, outputs 16.
- 사다리 3칸 각각 "할 수 있는 것 + 한계"를 정직하게 그음. 깨끗한 정지점.
- README 최신화 완료: AX서사+한shot흐름+가치사다리(한계포함)+결과+ML스택+ECOD negative+두 확장축(세로 사다리/가로 스위트).
- 재개 시 후보: (A)README/로드맵에 사다리 반영 (B)예측 칸 (C)SHAP (D')스위트 ③예지보전.
### 다음 후보 (택1)
- (A) README/로드맵에 가치사다리+인과 한계 반영(정직 경계=셀링포인트). (B) 예측 칸(드리프트 조기경보).
- (C) SHAP 업그레이드. (D) 여기서 정리/멈춤 — 매우 깊은 마일스톤(사다리 3칸+각 칸 한계 명시).

---

## 2026-06-23 (이어서) — 예측 칸: conformal test martingale 드리프트 조기경보 (완료)

조사 후 예측 칸 채택 이유: 검출 칸의 split-conformal p-value 스트림을 그대로 재사용 → 새 모델 0개,
"오탐율<=α 보장"이 "오경보율<=1/c 보장"으로 한 칸 상승(같은 measured-trust 척추), 미사용 35K
unlabeled가 자연스러운 무대. 선행: Vovk 2021 "Retrain or not retrain"(CTM), 2025-11 사출성형
드리프트+증분학습 논문(궤도 일치, 우리 차별점=경보 신뢰성 보장).

### 구현 → src/drift.py(54줄, SimpleJumper), eda/13_drift.py, outputs/13_drift.txt + figures/drift_martingale.png
- Simple Jumper [verified, arxiv 2102.10439]: 자본 3계좌 C_{-1,0,+1} 초기 1/3, betting f_eps(p)=1+eps(p-0.5),
  jumping rate J=0.01. 매 스텝 ①재분배 (1-J)C+(J/3)ΣC ②베팅 ③S_n=ΣC. Ville: P(sup S_n>=c)<=1/c.
- **A) 검증(보장):** 교환가능 정상 스트림 300회×1000shot → 오경보율(sup S_n>=100) **0.7% <= Ville 상한 1%** OK.
  S_n 최대 분포 중앙값 2.13/95%분위 19.83. = 정상이면 자본 1 근처 = 보장된 침묵. (conformal 오탐율 검증과 동일 패턴)
- **C) 검정력(lead-time):** 정상 500 후 Mold_Temperature_4(가스 시그니처) 점진 shift(200-shot 램프) 주입.
  **lead-time 1σ→146 / 2σ→92 / 3σ→74 shot, onset 전 헛경보 0건.** shift↑일수록 빨리 발산(강도-lead trade-off),
  램프 ~절반 지점에서 조기 포착. 약한 1σ는 천장(검출 recall 천장과 같은 정직 경계).
- **[검증결과/함정] B 실스트림 1차 시도 실패→정정:** unlabeled_cn7은 **열별 전역표준화(sd=1.0)**, supervised는
  **원단위**(Cycle 59.5·Barrel ~275). cross-file Mahalanobis 스코어링 시 p<0.05가 100%(전부 outlier)=전처리
  불일치지 드리프트 아님. unlabeled 표준화 파라미터 미상→원단위 복원 불가. **정정: unlabeled 한 파일 내 자기일관
  감시** — 초기 4000 shot=참조(in-control 가정), 이후 31,239 감시. p<0.05=3.4%(정상범위), **첫 변화점 @생산 shot
  4,015**(참조 종료 15shot 후), 에피소드 996개. first2000 vs last2000 평균차 Hopper_Temp+3.2σ·Barrel+2.3σ
  = 실 생산 드리프트 실재(머신 warm-up/조건 이동 추정). 한계: 참조 in-control은 라벨없는 가정.
- 가치사다리: 서술→[진단✅]→[처방✅]→[인과✅robust]→**[예측✅ 드리프트]**→자율. 예측 칸도 "보장+측정+정직한 천장" 유지.
- [정직한 한계] calib=검증스트림 같은 표본→super-uniform(보수적). 고정 calib 공유→엄밀 IID 아님(연속스코어라 영향 작음).
  단일라인 점진 드리프트만 주입.

### ■ 마일스톤 정지 (D 선택, 2026-06-23) — 가치사다리 4칸(진단·처방·인과·예측) 완료
- **산출물:** src/ 8모듈 ~350줄(conformal·detector·grounding·copilot·vanilla·prescribe·causal·drift),
  eda/01~13(재현 스크립트 13종), outputs 13 txt + figures 5(hist·scatter·pca·anomaly_scores·drift_martingale).
- **사다리:** 서술 → 진단✅ → 처방✅ → 인과✅robust → 예측✅ → (자율=빈칸). 각 칸 "할 수 있는 것 + 정직한 한계" 명시.
- **통일 척추 = measured trust:** 검출 오탐율≤α → 예측 오경보율≤1/c, 같은 종류의 유한표본 보장이 사다리 두 칸에서 반복.
  엔진 有/無 대조(우리 8% vs vanilla 68% confidently-wrong)로 정직성을 숫자로 증명. 거대기업이 안 하는 빈자리.
- **정직한 천장 모음:** 검출 recall 58% / 처방 상관 recourse 한계 / 인과 단일라인 관측 천장 / 예측 약한(1σ) 드리프트 천장.
  전부 측정해서 명시 — 자율 함부로 약속 안 함. ECOD는 측정해서 떨굼(negative result).
- **깨끗한 정지점.** README·worklog·메모리 동기 완료. 재개 후보: (A)bounded 자율(에스컬레이션 명시화) (B)SHAP
  (C)스위트 ③예지보전(가로축) (D)정지 유지.

---

## 2026-06-23 (이어서) — 코드 리뷰 + 평가 신뢰구간 (정지 후 보강)

### (A) 코드 품질 리뷰 — ha-review 방법론 수동 적용
- **ha-review 스킬은 실행 불가:** prepare가 harness-plan.md(HarnessAI 스캐폴딩) 요구 → "/ha-init 먼저"로 거부.
  리서치 프로젝트라 ha-init 개조 비추천 → 스킬의 검사항목(보안훅7+ai-slop+insecure-defaults+sharp-edges)을 수동 적용.
  (prepare git diff 위해 molding-copilot/에 git init + 스냅샷 커밋 f6ca884 1개 남김.)
- **결과: APPROVE, BLOCK 0건.** 순수 수치 연산 코드라 공격면 사실상 0(시크릿/SQL/auth/명령주입 없음).
- note 3건(전부 선택 개선): (1) vanilla.py:37 광범위 except — 단 baseline 대조군 의도설계(파싱실패=punt 측정, FP).
  (2) detector.py:9 SIGNATURE 미사용 import. (3) detector.fit fit/calib 분리가 docstring만·코드 미강제(겹치면 conformal 보장 조용히 깨짐) → assert 권장.

### (B) 평가 신뢰구간 → eda/14_bootstrap_ci.py, outputs/14_bootstrap_ci.txt
- 브랜드("양성 19개→분산 정직")에 정작 CI가 없던 헤드라인 숫자에 Wilson 95%CI + recall 부트스트랩(5000회) 부여. LLM 재실행 없이 저장 카운트로.
- **결과 [verified]:** 검출 recall 57.9% **CI[36.3, 76.9]**(부트스트랩 [36.8,78.9] 일치) — 매우 넓음=n19 현실.
  가스 53.8%[29,77]·미성형 66.7%[30,90]. 정상 오탐율 5.4%[4.4,6.6]=좁음(보장 견고). 모드적중 81.8%[52,95].
  **CW: 우리 7.5%[2.6,19.9] vs vanilla 67.5%[52,80], 차이 60%p CI[43,77]p 0 미포함 → 통계적 유의.**
- **메시지:** "어디는 확실/어디는 불확실"을 분리 — recall 점추정은 과신 금지(넓음), but 핵심 주장(CW 60%p 감소)은
  유의성으로 방어됨. = measured-trust 브랜드와 정합. eda/14종, outputs 14 txt로 확장.

### (C) 두 번째 실데이터 RG3로 일반화 검증 → eda/15_rg3_generalize.py, outputs/15_rg3_generalize.txt
- 동기: 사용자 "실제 자료로 테스트 가능?" → 디스크에 RG3 부품군 실데이터 존재(다운로드 불필요). RG3=CN7과
  **같은 기계(650톤 우진2호기), feature 24개 중 23개 동일**, labeled 1,256/진짜결함 32(가스22+미성형10)/cold-start 0.
  시그니처 변수(Mold_Temperature 등) 전부 RG3에 살아있음 → 혼동요인 없는 깨끗한 일반화 테스트.
- **정직한 혼합 결과 [verified]:**
  - **① 보장은 일반화 ✅:** RG3 정상 오탐율 4.9%≤5%(CN7 5.4%와 동일 궤도). conformal 분포가정無 → 라인 무관 보장. measured-trust 메커니즘 전이됨.
  - **② 검출력은 일반화 안 됨 ❌(코드버그 아님):** RG3 recall 12.5%(4/32) vs CN7 57.9%. 원인=RG3 결함이 공정변수상 거의 안 보임
    (가스 max z +0.7σ·미성형 +1.0σ; CN7은 +3~4σ였음). 같은 기계·변수인데도 5%FPR 게이트를 넘을 만큼 안 벗어남.
  - **③ 시그니처 부품 특수적 ❌:** CN7 가스=금형온도, **RG3 가스=금형온도 무관**(top=Injection_Time/Screw_RPM ~0.7σ).
    CN7 시그니처 RG3 적용 모드적중 2/4(가스0/2,미성형2/2,n작음→분리도로 판단). → **"약한 순환성" 한계가 실데이터로 확인**=시그니처 라인별 재유도 필요.
- **함의:** 실패 아님 = measured-trust 철학의 정확한 작동. **주장한 보장은 전이/과장 안 한 성능은 라인 의존을 정직히 인정/의심한 시그니처 한계는 실증.**
  → "범용 공장 copilot 안 짓고 단일공정 depth" 전략이 실증 정당화(같은 기계서조차 부품간 일반화 어려움). 일반화를 가정 않고 *측정*한 것이 가치. eda/15종.

### (D) 검출 강화 Lever 1 — conformal score를 지도모델로 교체 → eda/16, src/detector.py 통합
- 동기: 사용자 "불량 잡는 거 더 올리자". 내부단서=지도 LR ROC0.985(비지도 conformal 58% 대비)→검출기 약함((가))이 큼.
- 방법: conformal 보장은 검출기 무관(ECOD로 증명)→nonconformity score를 [Mahalanobis 거리]→[지도모델 불량확률]로 교체.
  19양성 과적합 방지 LOO 교차적합(각 결함은 자기뺀 모델로 채점). calib/test 정상 학습 미포함→오탐율 보장 유지.
- **결과 [verified, outputs/16]: recall 58%→지도LR 84%→지도GBM 89%, 오탐율 4.6%≤5% OK(보장 유지).**
  GBM 모드별 가스85%(11/13)·미성형100%(6/6). 17/19 검출, 2개만 놓침(=z<1σ 데이터한계(나)에 근접). 부트스트랩CI GBM[74,100].
  → **(가)검출기 약함이 지배적이었음 입증.** 데이터에 신호 있었고 비지도가 못 끌어낸 것.
- **옵션1 채택: src/detector.py 지도 게이트 통합** — fit(X_fit,X_calib,X_defects=None): 라벨 주면 GBM 게이트(불량확률 score),
  없으면 비지도 Mahalanobis(backward compat). 설명(z-편차 grounding)은 모드 무관 동일. 'mahalanobis'키→'score', 'gate_mode' 추가.
  스모크 검증: 지도 게이트 결함 p=0.001/score=0.947 발화·정상 p=0.937 기권. eda/07 비지도 backward compat OK.
- **trade-off(정직):** 이제 검출에 결함 라벨 필요(완전 비지도 장점 포기). RG3서 봤듯 라인 바뀌면 그 라인 라벨로 재학습 필요(전이 미보장).
  비지도 게이트는 fallback으로 남김(라벨 없는 라인). eda/16종.

### (E) 검출 강화 ① 하이브리드(지도+비지도) — novel 결함 안전망 → eda/17_hybrid_ensemble.py, outputs/17
- 동기: 지도 게이트의 새 약점 = 라벨된 결함모드만 앎(처음 보는 결함=unknown unknown에 약함). Mahalanobis를 안전망으로.
- **(1) 점수 블렌딩 실패(negative):** 지도확률+Mahalanobis를 표준화 후 합/최대로 결합→단일 conformal 게이트.
  sum=79%·max=68% < 지도 89%. 원인=결합 null 분포가 부풀어 conformal 임계↑→지도가 잡던 것까지 놓침. 블렌딩 폐기.
- **(2) 합집합(Bonferroni α분할) 채택:** 지도 게이트(α1=0.04) OR Mahalanobis 게이트(α2=0.01), 합집합 오탐율≤α.
  각 게이트가 자기 null로 임계→지도 recall 보존. **결과[verified]: A) 합집합 89%(오탐율5.1%) = 지도-only 89% 유지.**
- **B) novel-모드 leave-out(★ 핵심):** 한 모드 통째로 빼고 학습. 가스 novel: 지도-only 15%(무너짐)·Maha(0.01) 15%·합집합 15%·
  **[다이얼업] Maha 풀예산(0.05) 54%.** 미성형 novel: 전부 67%(가스 학습모델도 미성형 부분포착-시그니처 겹침).
- **★핵심 발견(정직):** 고정 FPR 예산서 **known recall(지도) vs novel coverage(Maha)는 경합.** 89% known 유지하려 지도에 0.04
  주면 안전망 0.01=약함(novel 15%). novel 강하게(54%) 하려면 FPR 더 써야. → **안전망은 공짜 아닌 '다이얼'**(운영자가
  미지결함 공포 vs 오경보로 α2 택). 블렌딩 아닌 별도 escalation 채널이 옳음. z<1σ는 어느 예산서도 불검출(데이터 한계).
- 미반영: detector.py 듀얼게이트 배선은 보류(설계 선택). 현재 detector는 단일 지도/비지도 게이트. eda/17종.

### ■ 정지 — 검출 닫고 포트폴리오 consolidation (2026-06-23)
- 판단: 검출은 commodity(우리 논지)인데 Lever1/① 강화로 충분히 입증(58→89%, 한계까지 측정). 남은 2개는 데이터 한계라
  모델로 못 풂 → 추가 검출작업 기대수익 ≈0. "있는 깊이를 보여주는" 게 recall 한 점보다 값짐(대표작/대외 목적 확인됨).
- **SHOWCASE.md 작성:** 평가자(채용/대외) 관점 1장 — 논지 / 증명 역량(conformal·eval·인과·정직ML) / ★moat=레이어3
  (CW 7.5% vs 67.5%, 차이 60%p CI[43,77] 유의) / 가치사다리 4칸 성과+한계 / negative result 자산 / 재현.
  README(제품문서)·worklog(일지)와 별개. 검출 아닌 정직성 측정을 전면에.
- 다음 후보: PDF 렌더(make-pdf) / 히어로 그림 / 레이어3 eval 강화(RAGAS·LLM-judge) / cross-line 전이 / 정지 유지.

### ■ Streamlit 대시보드 — "진짜 프로그램처럼" 시각화 (2026-06-23)
- 동기: 사용자 "시각화까지, 진짜 프로그램처럼". 환경 확인=streamlit·plotly·matplotlib·ollama(gemma3:4b) 보유.
- **app.py + demo_core.py**: 검증/렌더 분리(코어=streamlit무관 테스트가능, UI=app). 코어 스모크 통과(지도/비지도×결함/정상×그림).
- 기능: 한 shot이 파이프라인 흐름을 실시간으로 — ①게이트 판정(이상/모른다)+conformal p-value 게이지 ②변수 z-편차
  막대(시그니처 색) ③근거 테이블(실측 인용) ④처방(lever/symptom) ⑤copilot 자연어(gemma3:4b, 버튼). 
  **★게이트 토글(지도/비지도)**: 같은 결함을 비지도는 "모른다"·지도는 잡는 걸 눈으로(Lever-1 인터랙티브). 탭2=측정된신뢰(CW 7.5 vs 67.5 막대+recall/오탐율/RG3 메트릭).
- 정직성: 데모 정상은 held-out(fit/calib 미포함=누출X). 헤드라인 숫자는 outputs서 측정한 상수(출처 eda 명시). 검증: py_compile OK + streamlit 헤드리스 부팅 HTTP200 무에러.
- 실행: `streamlit run app.py`. README 실행법 추가. src 8모듈 유지(+app.py·demo_core.py 2개).

### ■ Django + TypeScript 웹앱 — "더 전문적으로" 재구축 (2026-06-23)
- 동기: 사용자 "streamlit말고 typescript django로, 더 전문적으로". + **HarnessAI 코드스타일 준수 요청.**
- HarnessAI 컨벤션 적용(agent/harness/profiles): _base + fastapi(백엔드 참조) + react-vite(프론트).
  - 백엔드(webapp/, Django 6 + DRF): 뷰는 **service 레이어 경유**(직접 로직 금지), **응답 camelCase**, logger(print 금지),
    에러 래퍼 {error,code}, **TypedDict로 경계 타이핑**(Any 금지), DB 미사용(GET JSON만). service.py가 src/ 파이프라인 재사용(lru_cache).
    엔드포인트 /api/{shots,diagnose,trust}. **테스트 4개 통과**(SimpleTestCase, TDD 정신).
  - 프론트(frontend/, React+TS+Vite+Tailwind): **Zustand 단일 store**(action서 axios 직접, React Query 금지),
    shared/{api,store,types,components} + containers/{diagnosis,trust} 구조, **CVA**(VerdictCard 변형), 클래스 style 상수화,
    type="button", any/console.log 금지. **차트는 CSS/SVG 직접 구현**(화이트리스트에 charting 없음 → 준수). tsc 0 errors.
  - Vite 프록시 /api→Django(8000) = CORS 우회. 정직성: 데모 정상 held-out, trust 숫자는 측정 상수.
- **검증(browse 캡처): 양 탭 정상.** 진단탭=판정카드+p-value미터(0.002)+z막대(금형온도+3σ 빨강)+근거표+처방. 신뢰탭=CW 7.5 vs 67.5 막대+메트릭.
- **함정 2개 잡음:** ①DRF가 contenttypes/auth 앱 요구(INSTALLED_APPS 추가). ②--noreload로 띄운 뒤 trust_metrics camelCase 수정분 미반영→프론트 crash(blank)→Django 재기동으로 해결.
- 편차(정직): npm(pnpm 대신)·차트 직접구현(charting 미화이트리스트)·테스트는 핵심만. 실행: Django `python manage.py runserver` + Vite `npm run dev`.
- 산출: webapp/(Django) + frontend/(TS). Streamlit(app.py)는 경량 데모로 잔존. src/ ML 8모듈은 양쪽 공유.

### ■ 프론트 재디자인 + Streamlit 완전 폐기 (2026-06-23)
- 동기: 사용자 "스트림릿 버리고, 디자인 레퍼런스 찾아 다시 — 지금은 Streamlit과 차이 없다(평범)". 레퍼런스 제공(이미지2장
  =모던 헬스 대시보드 'AI-BL' 라임그린 톤 + Vercel URL).
- **디자인 시스템 추출·적용:** 라이트 배경(#ededf0) + **흰색 라운드 카드(22px)+소프트 섀도** + **라임 액센트(#c4ec3f)** +
  near-black 굵은 타이포(Inter) + pill 배지 + **좌측 아이콘 레일**(active=라임 원) + 상단 pill 클러스터.
  **색 의미 판단:** 레퍼런스 그린=긍정인데 우리는 이상=경보 → 라임은 브랜드/활성 크롬에만, **이상 판정=강한 다크 카드+레드닷**,
  z막대는 의미색(가스 빨강/미성형 파랑) 유지. tailwind 토큰화(ink/accent/gas/short, rounded-card, shadow-card).
- 데이터 배선(store/api/types) 불변, **스타일·레이아웃만** 재작성(App+5컴포넌트). lucide-react 아이콘. tsc 0 errors.
- 함정: tailwind.config에 ink 색 추가 전 Vite 기동→config stale로 `text-ink` 없음 500 → Vite 재기동으로 해결.
- **browse 재캡처: 양 탭 레퍼런스 톤 정확 반영 확인.** 진단=다크 판정카드+라임 모드+빨강 p-value+라운드 z막대+라임 처방카드.
  신뢰=우리 7.5%(라임)/vanilla 67.5%(블랙) 대비막대+메트릭 카드.
- **Streamlit 폐기:** app.py 삭제. demo_core.py는 백엔드가 쓰는 파이프라인 코어라 유지하되 plotly fig 함수(streamlit 전용) 제거(dead code).
  데모 정상 held-out · 4 django 테스트 통과 재확인. 산출: webapp/ + frontend/ (Streamlit 없음).

### ■ 코드/디자인 평가 + 디자인 3수정 + 데모 누설 수정 (2026-06-24)
- **코드 평가(사용자 "코드 직접 보고 평가"):** src/·webapp/·frontend 정독. 강점=conformal p-value 공식 정확
  (conformal.py:24)·score무관 게이트(detector.py)·SimpleJumper Vovk일치(drift.py)·service 레이어 경계 깔끔.
  발견한 문제 3건: **①지도 모드 데모 결함 in-sample 누설**(demo_core.build의 Xdef를 학습+데모에 동시 사용,
  정상만 held-out·결함은 아님) / ②`is_anomaly=True`인데 evidence=[]·mode=None 가능(지도 게이트 발화하나 |z|<2)
  / ③설명 로직 이원화(grounding.py GroundedExplainer z=3.0 런타임 dead vs detector.diagnose 인라인 z=2.0).
- **디자인 평가(browse 실렌더 캡처) + 3수정 적용:** 진단탭은 "디자이너급", 신뢰탭·일부 크롬은 AI티.
  ① 장식 크롬 제거(LIVE pill·🔔벨·라임✱ 배지=기능없는 빌려온 크롬) + 의미없는 타이틀 "대시보드"→"사출성형 결함진단"+서브타이틀.
  ② 신뢰탭 빈공간 채움(막대 h-56→h-44 + 0% 기준선 + 중앙 "−60%p 오진감소" 동적 델타 + 기준선 캡션 + 라벨 정렬).
  ③ p-value 카드 빈 하단을 mt-auto 해석 푸터(게이트 발화/기권 설명)로 채워 우측 z막대 카드와 바닥 정렬.
  tsc 0err · 콘솔 에러 0 · browse 재캡처 확인. (미수정: gate pill ⌄ 쉐브론 어포던스 불일치=범위 밖.)
- **★데모 누설 수정(코드리뷰 ① — LOO 교차적합):** 사용자 "다음 단계"로 선택. measured-trust 무결성 직결.
  - 수정: `demo_core.build(gate, exclude_idx=None)` — 지도 모드서 지정 결함을 학습 제외. `service.diagnose`가
    지도 게이트로 데모 결함 진단 시 `_pipeline_loo(idx)`(lru_cache32)로 그 shot 뺀 GBM 재적합. 비지도는 결함 미학습→불필요.
  - **검증(eda/18_demo_loo.py, outputs/18) [verified]:** 데모 결함 19개 in-sample vs LOO — **평균 p 0.0011→0.0146(13배),
    검출 19/19(100%)→18/19(95%).** 가스 1건 p 0.0017→0.1552로 "이상"→"기권"(그 shot 학습 포함시에만 잡혔던 누설검출).
  - 라이브 반영(Django 재기동): 첫 가스 진단 p 0.012(LOO), 처방 what-if 0.012→0.026 일관. 4 django 테스트 OK.
  - 짚음: 데모 LOO 18/19(95%) vs README/eda16 보고 17/19(89%) 소차 = 정상 split/설정 차이(19표본 분산 범위, 같은 LOO 방법론).

### ■ 코드리뷰 ②③ 정리 (2026-06-24, A)
- **③ 설명 로직 일원화:** 모드추정이 두 곳(grounding 단순다수결 / detector |z|가중)에 중복 + z-임계 분기(2.0 vs 3.0)였음.
  - grounding.py에 **공유 `infer_mode(evidence)`**(|z|가중, 우월 버전) + 의도 명시 상수 `Z_CITE_GATED=2.0`(게이트 後 인용)·
    `Z_CITE_STANDALONE=3.0`(게이트 無 z가 게이트 역할). detector·GroundedExplainer 둘 다 `infer_mode` 호출 = 모드추정 단일 출처.
    z-임계 차이는 **삭제 아니라 "게이트 有/無"로 문서화**(우연 아닌 의도). detector의 미사용 `_VAR2MODE` import 제거.
  - 검증: eda/05 **적중 13/15=87% 그대로**(가스9/11·미성형4/4 — 다수결→가중이 이 결함들엔 동일) → worklog/README 87% 유효.
    eda/16·demo_core 스모크·4 django 테스트 OK. (백엔드 출력 불변 = z임계 2.0 동일·로직 동치 → Django 재기동 불요.)
- **② 빈-근거 이상판정 fallback:** 지도 게이트 발화하나 단일변수 |z|<2σ면 evidence=[]·mode=None(백엔드는 이미 정직히 빈값).
  프론트 처리: VerdictCard "모드 미상 · 단일변수 근거 약함(다변량 패턴)", EvidenceTable "임계 넘는 변수 없음 — 다변량 조합 발화, 인용 안 지어냄".
  데모 결함은 항상 z>2라 실발생X = 방어적 분기(정합성 구멍 차단). tsc 0err, browse 재캡처로 정상경로 회귀 없음 확인.

### ■ README 데이터출처·재현 정리 + 스크린샷 임베드 (2026-06-24)
- 사용자: "측정값 정리 + 데이터 출처 사이트 안내(원본 올리면 안 됨) + README 주의점".
- **데이터 출처 검증(WebSearch/WebFetch):** [verified] KAMP 인공지능 제조 플랫폼(중소벤처기업부, 운영 KAIST,
  kamp@kaist.ac.kr) www.kamp-ai.kr 의 **사출성형 제조AI데이터셋**. 무료 회원가입 후 다운로드, 모델 상업활용 가능하나
  **원본 데이터셋 재배포 제한**. → README에 "데이터 출처·재현" 섹션 신설(출처+재배포제한+3단계 재현절차). data/ 는
  이미 .gitignore(256MB, git 미트래킹 확인). git 트래킹 42파일에 data/·.env·시크릿 0건.
- **신선도 정정:** eda 17→18종, 구조에 webapp/frontend, 프론트 포트 "5173(점유 시 5174)".
- **requirements.txt 신설:** 실제 import + 설치버전 고정(numpy2.2.6/pandas2.3.3/scikit-learn1.7.2/scipy1.16.3/
  lingam1.12.2/pyod3.6.1/ollama0.6.1/Django6.0.6/DRF3.17.1/cors4.9.0). 재현성 공백 메움.
- **스크린샷 임베드:** docs/dashboard-{diagnosis,trust}.png(browse 캡처본 repo로 복사) → README 히어로(진단,p0.012 LOO)+§4(신뢰탭 CW대조).
- README 작성 주의점 조언: 숫자엔 출처+CI(measured-trust 자기정합), 재현성(데이터 받는법+버전고정), 30초 히어로, 라이선스 정직성, 과장어 금지.
