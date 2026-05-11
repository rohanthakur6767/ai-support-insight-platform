from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TicketIn(BaseModel):
    ticket_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    customer_id: Optional[str] = None
    channel: Optional[str] = Field(default="email")
    message: str
    agent_reply: Optional[str] = None
    product: Optional[str] = None
    order_value: Optional[float] = 0.0
    customer_country: Optional[str] = None
    resolution_status: Optional[str] = "open"


class TicketOut(TicketIn):
    id: int
    category: Optional[str] = None
    category_confidence: Optional[float] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    frustration_level: Optional[int] = None
    issue_cluster: Optional[int] = None
    suggested_reply: Optional[str] = None
    processed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    received: int
    processed: int
    skipped: int


class CategoryCount(BaseModel):
    category: str
    count: int
    pct: float


class SentimentTrendPoint(BaseModel):
    date: str
    positive: int
    neutral: int
    negative: int


class TopIssue(BaseModel):
    cluster_id: int
    label: str
    count: int
    avg_order_value: float
    sample_messages: list[str]


class InsightsSummary(BaseModel):
    total_tickets: int
    by_category: list[CategoryCount]
    sentiment_trend: list[SentimentTrendPoint]
    top_issues: list[TopIssue]
    revenue_at_risk: float
    avg_frustration: float


class ReplyRequest(BaseModel):
    message: str
    category: Optional[str] = None
    customer_country: Optional[str] = None
    product: Optional[str] = None


class ReplyResponse(BaseModel):
    suggested_reply: str
    category: str
    sentiment: str
    frustration_level: int
