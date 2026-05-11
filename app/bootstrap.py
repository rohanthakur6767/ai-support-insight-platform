"""First-boot bootstrap.

Render's free tier has no persistent disk, so each cold start lands on a fresh
container. If the SQLite DB is empty (or missing), we synthesise a 5k-ticket
dataset and run the pipeline before the API starts serving requests. This costs
~30–60 s on the first boot and is then a no-op until the container is recycled.

Set SEED_ON_BOOT=0 to skip this (e.g., when running on infra with persistent
storage and a real seeded DB).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sqlalchemy import func, select

from app.config import PROJECT_ROOT, get_settings
from app.data.synthesize import generate_dataset
from app.models.db import Ticket, init_db, session_scope

logger = logging.getLogger(__name__)


def seed_if_empty() -> None:
    if os.environ.get("SEED_ON_BOOT", "1") in ("0", "false", "False"):
        logger.info("SEED_ON_BOOT disabled; skipping seed")
        return

    init_db()

    with session_scope() as s:
        count = s.execute(select(func.count(Ticket.id))).scalar_one() or 0

    if count > 0:
        logger.info("DB already populated (%d tickets); skipping seed", count)
        return

    n = int(os.environ.get("SEED_N", "5000"))
    csv_path = Path(get_settings().db_url.replace("sqlite:///", "")).parent / "tickets.csv"
    if not csv_path.exists():
        logger.info("Generating %d synthetic tickets at %s", n, csv_path)
        generate_dataset(csv_path, n=n)

    logger.info("Running pipeline on %s ...", csv_path)
    # Import here to avoid triggering heavy ML imports for callers that just want init_db.
    from app.pipeline.runner import process_csv

    summary = process_csv(str(csv_path))
    logger.info("Seed complete: %s", summary)
