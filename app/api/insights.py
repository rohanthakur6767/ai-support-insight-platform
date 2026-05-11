"""Insight aggregation endpoints — the business-facing analytics layer."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Query
from sqlalchemy import desc, func, select

from app.models.db import Ticket, session_scope
from app.models.schemas import (
    CategoryCount,
    InsightsSummary,
    SentimentTrendPoint,
    TopIssue,
)

router = APIRouter(prefix="/insights", tags=["insights"])


def _date_window(days: int) -> tuple[datetime, datetime]:
    end = datetime.utcnow()
    return end - timedelta(days=days), end


@router.get("/summary", response_model=InsightsSummary)
def summary(days: int = Query(30, ge=1, le=365)):
    start, end = _date_window(days)
    with session_scope() as s:
        total = s.execute(
            select(func.count(Ticket.id)).where(Ticket.timestamp >= start)
        ).scalar_one()

        # by category
        cat_rows = s.execute(
            select(Ticket.category, func.count(Ticket.id))
            .where(Ticket.timestamp >= start)
            .group_by(Ticket.category)
            .order_by(desc(func.count(Ticket.id)))
        ).all()
        by_category = [
            CategoryCount(category=c or "Unknown", count=n, pct=round(100 * n / max(total, 1), 1))
            for c, n in cat_rows
        ]

        # sentiment trend (daily)
        trend_rows = s.execute(
            select(Ticket.timestamp, Ticket.sentiment).where(Ticket.timestamp >= start)
        ).all()
        buckets: dict[str, dict[str, int]] = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0})
        for ts, sent in trend_rows:
            day = ts.date().isoformat()
            if sent in ("positive", "neutral", "negative"):
                buckets[day][sent] += 1
        trend = [
            SentimentTrendPoint(date=d, **v)
            for d, v in sorted(buckets.items())
        ]

        # top issues — derived from the issue_cluster column populated by the pipeline.
        cluster_rows = s.execute(
            select(
                Ticket.issue_cluster,
                func.count(Ticket.id),
                func.avg(Ticket.order_value),
            )
            .where(Ticket.timestamp >= start)
            .where(Ticket.issue_cluster.is_not(None))
            .group_by(Ticket.issue_cluster)
            .order_by(desc(func.count(Ticket.id)))
            .limit(8)
        ).all()

        top_issues: list[TopIssue] = []
        for cid, n, avg_ov in cluster_rows:
            samples = list(
                s.execute(
                    select(Ticket.message)
                    .where(Ticket.issue_cluster == cid)
                    .where(Ticket.timestamp >= start)
                    .limit(3)
                ).scalars()
            )
            top_issues.append(
                TopIssue(
                    cluster_id=int(cid),
                    label=_short_label(samples[0]) if samples else f"Cluster {cid}",
                    count=int(n),
                    avg_order_value=float(avg_ov or 0.0),
                    sample_messages=samples,
                )
            )

        # revenue at risk = sum of order_value for unresolved + high-frustration tickets
        rev_at_risk = s.execute(
            select(func.coalesce(func.sum(Ticket.order_value), 0.0))
            .where(Ticket.timestamp >= start)
            .where(Ticket.resolution_status.in_(("open", "in_progress", "escalated")))
            .where(Ticket.frustration_level >= 2)
        ).scalar_one()

        avg_frustration = s.execute(
            select(func.coalesce(func.avg(Ticket.frustration_level), 0.0))
            .where(Ticket.timestamp >= start)
        ).scalar_one()

        return InsightsSummary(
            total_tickets=int(total),
            by_category=by_category,
            sentiment_trend=trend,
            top_issues=top_issues,
            revenue_at_risk=round(float(rev_at_risk), 2),
            avg_frustration=round(float(avg_frustration), 2),
        )


@router.get("/revenue-by-category")
def revenue_by_category(days: int = Query(30, ge=1, le=365)):
    start, _ = _date_window(days)
    with session_scope() as s:
        rows = s.execute(
            select(
                Ticket.category,
                func.count(Ticket.id),
                func.coalesce(func.sum(Ticket.order_value), 0.0),
                func.coalesce(func.avg(Ticket.frustration_level), 0.0),
            )
            .where(Ticket.timestamp >= start)
            .group_by(Ticket.category)
            .order_by(desc(func.sum(Ticket.order_value)))
        ).all()
        return [
            {
                "category": c or "Unknown",
                "tickets": int(n),
                "revenue_touched": round(float(rev), 2),
                "avg_frustration": round(float(fr), 2),
            }
            for c, n, rev, fr in rows
        ]


@router.get("/volume-by-day")
def volume_by_day(days: int = Query(30, ge=1, le=365), category: Optional[str] = None):
    start, _ = _date_window(days)
    with session_scope() as s:
        stmt = select(Ticket.timestamp, Ticket.category).where(Ticket.timestamp >= start)
        if category:
            stmt = stmt.where(Ticket.category == category)
        rows = s.execute(stmt).all()

    counts: dict[str, int] = defaultdict(int)
    for ts, _ in rows:
        counts[ts.date().isoformat()] += 1
    return [{"date": d, "count": c} for d, c in sorted(counts.items())]


def _short_label(msg: str, max_chars: int = 90) -> str:
    msg = (msg or "").strip()
    for sep in (". ", "? ", "! "):
        i = msg.find(sep)
        if 15 < i < max_chars:
            return msg[:i].strip().rstrip(",.!?")
    return msg[:max_chars].strip().rstrip(",.!?")
