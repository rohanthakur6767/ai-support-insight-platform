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

# Pre-seed the database + vector store at build time.
#
# We run the full pipeline here (Render's build VM has plenty of RAM/CPU) so
# the runtime container only needs to serve, not crunch. This is what lets us
# fit on a 512 MB free-tier instance — the heavy embed/classify/cluster pass
# is already done by the time the container boots.
ARG SEED_N=5000
RUN mkdir -p /app/data && \
    python -m scripts.generate_data --n ${SEED_N} --out /app/data/tickets.csv && \
    python -m scripts.run_pipeline --csv /app/data/tickets.csv && \
    rm /app/data/tickets.csv

ENV PORT=8000 \
    SEED_ON_BOOT=0
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Shell form so $PORT (injected by Render / Cloud Run / Fly) is expanded.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
