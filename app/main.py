"""FastAPI entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import insights as insights_api
from app.api import tickets as tickets_api
from app.bootstrap import seed_if_empty
from app.config import get_settings
from app.models.db import init_db


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    cfg = get_settings()
    _configure_logging(cfg.log_level)
    init_db()
    try:
        seed_if_empty()
    except Exception as exc:  # noqa: BLE001 - boot must not crash on a seed failure
        logging.getLogger(__name__).exception("Seed failed (continuing without data): %s", exc)
    logging.getLogger(__name__).info("API ready (model=%s)", cfg.groq_model)
    yield


app = FastAPI(
    title="AI Support Insight Platform",
    version="0.1.0",
    description=(
        "Ingest customer support tickets and surface actionable insights — categories, "
        "sentiment trends, top recurring issues, revenue exposure, and suggested replies."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tickets_api.router)
app.include_router(insights_api.router)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "ai-support-insight-platform",
        "docs": "/docs",
        "health": "/health",
    }
