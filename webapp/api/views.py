"""api/views.py — 읽기전용 JSON 엔드포인트. 로직은 service 레이어에 위임.
에러는 내부 상세를 숨기고 {error, code} 래퍼만 반환(_base §3)."""
import logging

from rest_framework.decorators import api_view
from rest_framework.response import Response

from . import service

logger = logging.getLogger("api")

_VALID_GATE = ("supervised", "mahalanobis")
_VALID_CAT = ("가스", "미성형", "정상")


@api_view(["GET"])
def shots(request):
    """GET /api/shots?cat=가스|미성형|정상 → [{idx, groundTruth}]."""
    cat = request.GET.get("cat", "가스")
    if cat not in _VALID_CAT:
        return Response({"error": "invalid cat", "code": "BAD_CAT"}, status=400)
    return Response(service.list_shots(cat))


@api_view(["GET"])
def diagnose(request):
    """GET /api/diagnose?idx=N&gate=supervised|mahalanobis → 진단 결과."""
    gate = request.GET.get("gate", "supervised")
    if gate not in _VALID_GATE:
        return Response({"error": "invalid gate", "code": "BAD_GATE"}, status=400)
    try:
        idx = int(request.GET.get("idx", ""))
    except ValueError:
        return Response({"error": "idx must be int", "code": "BAD_IDX"}, status=400)
    try:
        return Response(service.diagnose(idx, gate))
    except IndexError:
        return Response({"error": "idx out of range", "code": "IDX_RANGE"}, status=404)


@api_view(["GET"])
def trust(request):
    """GET /api/trust → 레이어3 측정 지표."""
    return Response(service.trust_metrics())
