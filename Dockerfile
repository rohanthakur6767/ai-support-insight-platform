# Multi-stage build: small final image, no build toolchain in the runtime layer.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/.hf_cache

WORKDIR /app

# System deps: build tools for any wheels lacking pre-built arm/x86 binaries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Warm the embedding model into the image so the first request isn't a cold download.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY app ./app
COPY dashboard ./dashboard
COPY scripts ./scripts

RUN mkdir -p /app/data

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Shell form so $PORT (injected by Render / Cloud Run / Fly) is expanded.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
