"""demo/precompute.py — HF Spaces 데모용 파생값 사전계산 (로컬 전용, data/ 필요).

KAMP 원본은 재배포 제한 → 데모에는 **절대 측정값을 일절 내보내지 않는다.** 각 shot 은
정상 대비 표준편차(z)와 판정값(conformal p, verdict)만 남기고, 관측값·정상 평균/표준편차는
버린다. 코파일럿·처방 문장도 z 기반으로만 생성해 원자료 숫자(℃ 등)가 새지 않게 한다.
결함 진단은 webapp 과 동일한 LOO 교차적합 = 보고 recall(eda/16)과 같은 out-of-sample 기준.
코파일럿 문장은 로컬 gemma3:4b 로 여기서 1회 생성해 JSON 에 고정한다.
실행: python demo/precompute.py   산출: demo/assets/shots.json
"""
import importlib.util
import json
import os
import sys

import numpy as np
import ollama

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import demo_core as core  # noqa: E402
from src.copilot import SYS  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "service", os.path.join(ROOT, "webapp", "api", "service.py"))
service = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(service)

N_NORMAL = 12  # held-out 정상에서 뽑는 데모 표본 (α=5% 기대 오탐 ~0.6개 — 있으면 그대로 노출)
SYS_ZONLY = SYS + (" 모든 수치는 정상 대비 표준편차(z, 표준편차 배수)로만 말하라. "
                   "절대 측정값(온도 ℃, 압력 등 원자료 숫자)은 데이터에 없으니 절대 언급하지 마라.")
ABSTAIN = ("근거 불충분: 측정 공정변수상 정상과 구분되지 않습니다. "
           "원인을 단정할 수 없습니다 (지어내지 않음).")


def copilot_zonly(evidence_z, mode):
    ev = {"mode": mode, "evidence": evidence_z}       # z 만 담긴 근거 — 절대값 통로 없음
    out = ollama.chat(model="gemma3:4b", options={"temperature": 0}, messages=[
        {"role": "system", "content": SYS_ZONLY},
        {"role": "user", "content": f"질문: 이 제품이 왜 불량인가?\nEVIDENCE:\n"
                                    f"{json.dumps(ev, ensure_ascii=False)}"}])
    return out["message"]["content"].strip()


def prescription_zonly(pres, zmap):
    if not pres.get("needed"):
        return None
    parts = []
    lever_vars = [s.split()[0] for s in pres["levers"]]      # "var 27.5→23.46 (..)" → var
    if lever_vars:
        parts.append("직접 조정(lever): " + ", ".join(
            f"{v} (정상 대비 {zmap.get(v, 0):+.1f}σ) → 정상 방향" for v in lever_vars))
    if pres["symptoms"]:
        parts.append(f"증상(symptom) {', '.join(pres['symptoms'])} 은 결과일 뿐 → "
                     f"상류 점검: {', '.join(pres['upstream'])}")
    verdict = "정상 복귀 예측" if pres["resolved"] else "정상 복귀 미확인 → 공정 검토 권고"
    return ("처방 — " + " | ".join(parts) +
            f"  ⇒ what-if(직접조정 기준): conformal p {pres['p_before']:.3f}→{pres['p_after']:.3f} ({verdict})")


def main():
    _, _, X, reason, demo_normal = service._pipeline("supervised")
    didx = [int(i) for i in np.where(np.isin(reason, ["가스", "미성형"]))[0]]
    nidx = [int(i) for i in np.random.default_rng(7).choice(demo_normal, N_NORMAL, replace=False)]

    shots = []
    for n, idx in enumerate(didx + nidx, 1):
        d = dict(service.diagnose(idx, "supervised"))
        zmap = {e["var"]: e["z"] for e in d["evidence"]}
        d["evidence"] = [{"var": e["var"], "z": e["z"]} for e in d["evidence"]]  # 절대값 폐기

        is_defect = idx in didx
        _, presc, Xp, *_ = (service._pipeline_loo(idx) if is_defect
                            else service._pipeline("supervised"))
        if d["isAnomaly"]:
            d["prescription"] = prescription_zonly(presc.prescribe(Xp.iloc[idx]), zmap)
            d["copilot"] = copilot_zonly(d["evidence"], d["mode"])
        else:
            d["prescription"] = None
            d["copilot"] = ABSTAIN
        d["id"] = f"S{n:02d}"                 # 원본 식별자(serial/timestamp)는 노출하지 않는다
        d.pop("score", None)                  # 게이트 점수도 안 내보낸다
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
