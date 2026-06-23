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
