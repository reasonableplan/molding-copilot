"""Django settings — 現答 API (읽기전용 JSON, DB 미사용).
ML 파이프라인을 노출하는 가벼운 백엔드라 인증/세션/DB 없이 최소 구성."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'dev-only-not-for-production')
DEBUG = True
ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'django.contrib.contenttypes',  # DRF가 import 시 요구 (DB 테이블은 미사용)
    'django.contrib.auth',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
]

ROOT_URLCONF = 'config.urls'
WSGI_APPLICATION = None

# GET JSON만 다루므로 DB 미접근. Django가 default 키를 요구해 메모리 sqlite로 둠(마이그레이션 안 함).
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}

# 개발: Vite 프론트(다른 포트)에서 호출 → CORS 허용.
CORS_ALLOW_ALL_ORIGINS = True

REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': ['rest_framework.renderers.JSONRenderer'],
    'DEFAULT_AUTHENTICATION_CLASSES': [],
    'DEFAULT_PERMISSION_CLASSES': ['rest_framework.permissions.AllowAny'],
}

STATIC_URL = 'static/'
USE_TZ = True
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
