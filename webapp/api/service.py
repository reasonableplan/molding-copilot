"""api/service.py — ML 파이프라인 ↔ HTTP 경계 서비스 레이어.

뷰(views.py)는 절대 src/ 파이프라인을 직접 만지지 않고 이 레이어만 호출한다(레이어 경계 엄수).
응답 키는 프론트(TypeScript) 정합을 위해 camelCase. 무거운 적합은 gate 별 1회 캐시.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Literal, TypedDict

# molding-copilot 루트를 path 에 (src/, demo_core 재사용). service.py = webapp/api/service.py → 3단계 상위.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import demo_core as core  # noqa: E402
from src.grounding import _VAR2MODE  # noqa: E402

Gate = Literal["supervised", "mahalanobis"]
Category = Literal["가스", "미성형", "정상"]
ZBAR_TOP = 8  # z-편차 막대에 보일 변수 개수


class ShotItem(TypedDict):
    idx: int
    groundTruth: str


class EvidenceItem(TypedDict):
    var: str
    observed: float
    normalMean: float
    normalSd: float
    z: float


class ZBar(TypedDict):
    var: str
    z: float
    mode: str | None  # '가스' | '미성형' | None — 프론트 색상 매핑용


class Diagnosis(TypedDict):
    verdict: str
    isAnomaly: bool
    pValue: float
    alpha: float
    score: float
    gateMode: str
    mode: str | None
    groundTruth: str
    evidence: list[EvidenceItem]
    zbars: list[ZBar]
    prescription: str | None


@lru_cache(maxsize=2)
def _pipeline(gate: Gate):
    """gate 별 (detector, prescriber, X, reason, demo_normal) 1회 적합 후 캐시."""
    return core.build(gate)


@lru_cache(maxsize=32)
def _pipeline_loo(idx: int):
    """지도 모드에서 데모 결함 shot(idx)을 그 shot 제외하고 재적합 — in-sample 누설 차단.
    eda/16의 LOO 교차적합과 동일 방법론(대시보드 진단 = 보고 recall과 같은 기준)."""
    return core.build("supervised", exclude_idx=idx)


def list_shots(cat: Category) -> list[ShotItem]:
    """카테고리(가스/미성형/정상)별 데모용 shot 인덱스 + 실제 라벨."""
    _, _, _, reason, demo_normal = _pipeline("supervised")
    indices = core.sample_indices(reason, demo_normal)[cat]
    return [{"idx": int(i), "groundTruth": str(reason[i])} for i in indices]


def diagnose(idx: int, gate: Gate) -> Diagnosis:
    """한 shot 을 파이프라인에 통과시켜 진단 결과(camelCase)를 반환."""
    det, presc, X, reason, _ = _pipeline(gate)
    # 지도 게이트로 '데모에 표시되는 결함'을 진단할 땐 그 shot이 학습셋에 포함되므로(누설),
    # 그 shot을 뺀 분류기로 재적합해 정직한 out-of-sample 진단을 낸다. 비지도는 결함 미학습 → 불필요.
    if gate == "supervised" and str(reason[idx]) in ("가스", "미성형"):
        det, presc, X, reason, _ = _pipeline_loo(idx)
    shot = X.iloc[idx]
    d = det.diagnose(shot)

    z = ((shot[det.cols] - det.mu) / det.sd).dropna()
    z = z.reindex(z.abs().sort_values(ascending=False).index).head(ZBAR_TOP)
    zbars: list[ZBar] = [{"var": v, "z": round(float(z[v]), 2), "mode": _VAR2MODE.get(v)}
                         for v in z.index]
    evidence: list[EvidenceItem] = [
        {"var": e["var"], "observed": e["observed"], "normalMean": e["normal_mean"],
         "normalSd": e["normal_sd"], "z": e["z"]} for e in d["evidence"]]

    return {
        "verdict": d["verdict"], "isAnomaly": bool(d["is_anomaly"]),
        "pValue": round(float(d["p_value"]), 4), "alpha": det.alpha,
        "score": round(float(d["score"]), 4), "gateMode": d["gate_mode"],
        "mode": d["mode"], "groundTruth": str(reason[idx]),
        "evidence": evidence, "zbars": zbars,
        "prescription": presc.prescribe(shot)["text"] if d["is_anomaly"] else None,
    }


def trust_metrics() -> dict[str, object]:
    """레이어3 핵심 지표(측정값 상수 — 출처는 outputs/ eda 스크립트). 키는 camelCase."""
    t = core.TRUST
    return {
        "recallUnsup": t["recall_unsup"], "recallSup": t["recall_sup"],
        "recallCi": t["recall_ci"], "cwOurs": t["cw_ours"], "cwVanilla": t["cw_vanilla"],
        "cwDiffCi": t["cw_diff_ci"], "fprGuarantee": t["fpr_guarantee"], "alpha": t["alpha"],
        "rg3Fpr": t["rg3_fpr"], "rg3Recall": t["rg3_recall"],
    }
