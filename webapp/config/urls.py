"""URL 라우팅 — 읽기전용 API 3개."""
from django.urls import path

from api import views

urlpatterns = [
    path("api/shots", views.shots),
    path("api/diagnose", views.diagnose),
    path("api/trust", views.trust),
]
