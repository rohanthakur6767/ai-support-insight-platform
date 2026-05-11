"""FastAPI entrypoint."""
from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import insights as insights_api
from app.api import tickets as tickets_api
from app.bootstrap import seed_if_empty
from app.config import get_settings
from app.models.db import init_db


# Tracks whether the first-boot seed has finished. Surfaced by /health.
_SEED_STATE = {"running": False, "done": False, "error": None}


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


def _run_seed_background() -> None:
    """Run seed_if_empty() in a daemon thread so it can't block the event loop."""
    log = logging.getLogger(__name__)
    _SEED_STATE["running"] = True
    try:
        seed_if_empty()
        _SEED_STATE["done"] = True
        log.info("Background seed complete")
    except Exception as exc:  # noqa: BLE001
        _SEED_STATE["error"] = str(exc)
        log.exception("Background seed failed: %s", exc)
    finally:
        _SEED_STATE["running"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    cfg = get_settings()
    _configure_logging(cfg.log_level)
    init_db()
    # Kick the seed off in a daemon thread so uvicorn opens its port immediately.
    # On platforms with a port-scan timeout (Render), this is what lets the deploy
    # succeed even though the first-boot seed takes ~30–60 s.
    threading.Thread(target=_run_seed_background, name="seed", daemon=True).start()
    logging.getLogger(__name__).info("API ready (model=%s); seed running in background", cfg.groq_model)
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
    return {
        "status": "ok",
        "seed": {
            "running": _SEED_STATE["running"],
            "done": _SEED_STATE["done"],
            "error": _SEED_STATE["error"],
        },
    }


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "ai-support-insight-platform",
        "docs": "/docs",
        "health": "/health",
    }
