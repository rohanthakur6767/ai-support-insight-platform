from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(64), unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    customer_id = Column(String(64), index=True)
    channel = Column(String(16), index=True)
    message = Column(Text, nullable=False)
    agent_reply = Column(Text)
    product = Column(String(128), index=True)
    order_value = Column(Float, default=0.0)
    customer_country = Column(String(64))
    resolution_status = Column(String(32), index=True)

    # Enrichment columns (filled by pipeline)
    category = Column(String(64), index=True)
    category_confidence = Column(Float)
    sentiment = Column(String(16), index=True)  # positive / neutral / negative
    sentiment_score = Column(Float)              # -1.0 .. 1.0
    frustration_level = Column(Integer, index=True)  # 0..4
    issue_cluster = Column(Integer, index=True)
    suggested_reply = Column(Text)
    processed_at = Column(DateTime)


_settings = get_settings()
_engine = create_engine(_settings.db_url, future=True)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(_engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
