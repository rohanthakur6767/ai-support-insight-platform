"""Ticket ingestion + retrieval endpoints."""
from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from sqlalchemy import desc, select

from app.models.db import Ticket, session_scope
from app.models.schemas import (
    ReplyRequest,
    ReplyResponse,
    TicketIn,
    TicketOut,
    UploadResponse,
)
from app.pipeline.classify import classify
from app.pipeline.embed import search
from app.pipeline.reply import suggest_reply
from app.pipeline.runner import process_records
from app.pipeline.sentiment import analyze

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("", response_model=UploadResponse, status_code=201)
def create_ticket(ticket: TicketIn):
    summary = process_records([ticket.model_dump()], do_cluster=False)
    return summary


@router.post("/bulk", response_model=UploadResponse, status_code=201)
def create_tickets_bulk(tickets: list[TicketIn]):
    return process_records([t.model_dump() for t in tickets])


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_csv(file: UploadFile = File(...)):
    """Accept a CSV file of tickets. Required column: `message`. Other fields optional."""
    if not file.filename or not file.filename.lower().endswith((".csv", ".tsv")):
        raise HTTPException(status_code=400, detail="Expected a .csv or .tsv file")
    content = await file.read()
    try:
        import pandas as pd  # lazy — heavy import

        sep = "\t" if file.filename.lower().endswith(".tsv") else ","
        df = pd.read_csv(io.BytesIO(content), sep=sep)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}") from exc

    if "message" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'message' column")

    return process_records(df.to_dict(orient="records"))


@router.get("", response_model=list[TicketOut])
def list_tickets(
    category: Optional[str] = None,
    sentiment: Optional[str] = None,
    min_frustration: Optional[int] = Query(None, ge=0, le=4),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    stmt = select(Ticket).order_by(desc(Ticket.timestamp)).limit(limit).offset(offset)
    if category:
        stmt = stmt.where(Ticket.category == category)
    if sentiment:
        stmt = stmt.where(Ticket.sentiment == sentiment)
    if min_frustration is not None:
        stmt = stmt.where(Ticket.frustration_level >= min_frustration)

    with session_scope() as s:
        rows = list(s.execute(stmt).scalars())
        return [TicketOut.model_validate(r) for r in rows]


@router.get("/{ticket_id}", response_model=TicketOut)
def get_ticket(ticket_id: str):
    with session_scope() as s:
        t = s.execute(select(Ticket).where(Ticket.ticket_id == ticket_id)).scalar_one_or_none()
        if not t:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return TicketOut.model_validate(t)


@router.get("/search/semantic")
def semantic_search(q: str = Query(..., min_length=2), k: int = Query(5, ge=1, le=25)):
    return {"query": q, "results": search(q, k=k)}


@router.post("/reply", response_model=ReplyResponse)
def generate_reply(req: ReplyRequest):
    cat, conf = classify(req.message) if not req.category else (req.category, 1.0)
    sent = analyze(req.message)
    reply = suggest_reply(req.message, category=cat, product=req.product)
    return ReplyResponse(
        suggested_reply=reply,
        category=cat,
        sentiment=sent["sentiment"],
        frustration_level=sent["frustration_level"],
    )
