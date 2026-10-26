# ---- builder: 의존성만 설치 (최종 이미지에는 pip/빌드도구가 안 남게 분리) ----
FROM python:3.14-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime: 설치된 패키지 + 앱 코드만 ----
FROM python:3.14-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
