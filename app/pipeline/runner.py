"""End-to-end pipeline orchestrator.

Stages (each idempotent):
  1. ingest  – load raw rows (csv / dataframe / dicts)
  2. clean   – normalise text, drop empties
  3. enrich  – classify category, score sentiment + frustration
  4. cluster – run KMeans on the corpus, assign cluster_id per ticket
  5. store   – upsert rows into SQLite + vectors into Chroma

Designed to be safe to re-run: same ticket_id won't be duplicated.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Iterable

from sqlalchemy import select

from app.models.db import SessionLocal, Ticket, init_db, session_scope
from app.pipeline.classify import classify_batch
from app.pipeline.clean import clean_text, is_valid
from app.pipeline.embed import upsert_vectors
from app.pipeline.issues import IssueCluster, cluster_issues
from app.pipeline.sentiment import analyze

logger = logging.getLogger(__name__)


def _ensure_ticket_id(row: dict, idx: int) -> str:
    tid = row.get("ticket_id")
    return tid or f"TCK-AUTO-{int(time.time() * 1000)}-{idx}"


def _parse_ts(value) -> datetime:
    if value is None or value == "":
        return datetime.utcnow()
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.utcnow()


def process_records(rows: Iterable[dict], do_cluster: bool = True) -> dict:
    """Run the full pipeline for a batch of raw ticket dicts.

    Returns a small summary dict (counts) suitable for an API response.
    """
    init_db()
    rows = list(rows)
    received = len(rows)
    if not rows:
        return {"received": 0, "processed": 0, "skipped": 0}

    # --- Stage 1+2: clean ---
    cleaned: list[dict] = []
    for i, raw in enumerate(rows):
        msg = clean_text(raw.get("message"))
        if not is_valid(msg):
            continue
        cleaned.append({**raw, "_message": msg, "_idx": i})
    skipped = received - len(cleaned)
    if not cleaned:
        return {"received": received, "processed": 0, "skipped": skipped}

    # --- Stage 3: enrich ---
    msgs = [c["_message"] for c in cleaned]
    categories = classify_batch(msgs)
    sentiments = [analyze(m) for m in msgs]

    enriched: list[dict] = []
    for c, (cat, conf), sent in zip(cleaned, categories, sentiments):
        enriched.append(
            {
                "ticket_id": _ensure_ticket_id(c, c["_idx"]),
                "timestamp": _parse_ts(c.get("timestamp")),
                "customer_id": c.get("customer_id"),
                "channel": c.get("channel") or "email",
                "message": c["_message"],
                "agent_reply": c.get("agent_reply") or "",
                "product": c.get("product"),
                "order_value": float(c.get("order_value") or 0.0),
                "customer_country": c.get("customer_country"),
                "resolution_status": c.get("resolution_status") or "open",
                "category": cat,
                "category_confidence": conf,
                "sentiment": sent["sentiment"],
                "sentiment_score": sent["sentiment_score"],
                "frustration_level": sent["frustration_level"],
                "processed_at": datetime.utcnow(),
            }
        )

    # --- Stage 4: cluster (optional, expensive on small batches so we no-op those) ---
    clusters: list[IssueCluster] = []
    if do_cluster and len(enriched) >= 24:
        clusters = cluster_issues(
            [e["ticket_id"] for e in enriched],
            [e["message"] for e in enriched],
            [e["order_value"] for e in enriched],
            [e["frustration_level"] for e in enriched],
        )
        # map ticket_id -> cluster_id
        cmap: dict[str, int] = {}
        for cl in clusters:
            for tid in cl.member_ticket_ids:
                cmap[tid] = cl.cluster_id
        for e in enriched:
            e["issue_cluster"] = cmap.get(e["ticket_id"])
    else:
        for e in enriched:
            e["issue_cluster"] = None

    # --- Stage 5: persist ---
    with session_scope() as s:
        existing = {
            t.ticket_id
            for t in s.execute(
                select(Ticket.ticket_id).where(Ticket.ticket_id.in_([e["ticket_id"] for e in enriched]))
            ).scalars()
        }
        for e in enriched:
            if e["ticket_id"] in existing:
                # update enrichment in place
                t = s.execute(select(Ticket).where(Ticket.ticket_id == e["ticket_id"])).scalar_one()
                for k, v in e.items():
                    setattr(t, k, v)
            else:
                s.add(Ticket(**e))

    # vector upsert (outside the SQL tx so a failure here doesn't roll back the DB write)
    upsert_vectors(
        ticket_ids=[e["ticket_id"] for e in enriched],
        texts=[e["message"] for e in enriched],
        metadatas=[
            {
                "category": e["category"],
                "sentiment": e["sentiment"],
                "frustration_level": e["frustration_level"],
                "channel": e["channel"] or "",
                "product": e["product"] or "",
                "timestamp": e["timestamp"].isoformat(timespec="seconds"),
            }
            for e in enriched
        ],
    )

    logger.info(
        "Pipeline run: received=%d processed=%d skipped=%d clusters=%d",
        received,
        len(enriched),
        skipped,
        len(clusters),
    )
    return {"received": received, "processed": len(enriched), "skipped": skipped}


def process_csv(path: str, do_cluster: bool = True) -> dict:
    import pandas as pd  # heavy — import on demand

    df = pd.read_csv(path)
    return process_records(df.to_dict(orient="records"), do_cluster=do_cluster)
