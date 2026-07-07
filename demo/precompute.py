"""demo/precompute.py — HF Spaces 데모용 파생값 사전계산 (로컬 전용, data/ 필요).

KAMP 원본 데이터는 재배포 불가 → 데모에는 진단 파생값(conformal p, z-편차, 처방,
사전생성 코파일럿 문장)만 내보낸다. 결함 진단은 webapp 과 동일한 LOO 교차적합
(service._pipeline_loo) = 보고 recall(eda/16)과 같은 out-of-sample 기준.
코파일럿 문장은 로컬 gemma3:4b(Ollama)로 여기서 1회 생성해 JSON 에 고정한다.
실행: python demo/precompute.py   산출: demo/assets/shots.json
"""
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import demo_core as core  # noqa: E402
from src.copilot import Copilot  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "service", os.path.join(ROOT, "webapp", "api", "service.py"))
service = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(service)

N_NORMAL = 12  # held-out 정상에서 뽑는 데모 표본 (α=5% 기대 오탐 ~0.6개 — 있으면 그대로 노출)


def copilot_text(idx: int, is_defect: bool) -> str:
    det = (service._pipeline_loo(idx) if is_defect else service._pipeline("supervised"))[0]
    X = service._pipeline("supervised")[2]
    return Copilot(det).ask(X.iloc[idx])["answer"]


def main():
    _, _, X, reason, demo_normal = service._pipeline("supervised")
    didx = [int(i) for i in np.where(np.isin(reason, ["가스", "미성형"]))[0]]
    nidx = [int(i) for i in np.random.default_rng(7).choice(demo_normal, N_NORMAL, replace=False)]

    shots = []
    for n, idx in enumerate(didx + nidx, 1):
        d = dict(service.diagnose(idx, "supervised"))
        d["id"] = f"S{n:02d}"  # 원본 식별자(serial/timestamp)는 노출하지 않는다
        d["copilot"] = copilot_text(idx, idx in didx)
        shots.append(d)
        print(f"{d['id']} {d['groundTruth']:3} -> {d['verdict']} (p={d['pValue']})")

    out = {"gate": "supervised", "alpha": 0.05,
           "trust": service.trust_metrics(), "shots": shots}
    dst = os.path.join(ROOT, "demo", "assets", "shots.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    detected = sum(s["isAnomaly"] for s in shots if s["groundTruth"] != "정상")
    fp = sum(s["isAnomaly"] for s in shots if s["groundTruth"] == "정상")
    print(f"wrote {dst}: shots {len(shots)} (defect {len(didx)} / normal {len(nidx)}), "
          f"defect detected {detected}/{len(didx)}, normal false-positive {fp}/{len(nidx)}")


if __name__ == "__main__":
    main()
