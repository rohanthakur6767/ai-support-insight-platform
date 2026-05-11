"""Smoke tests for the AI pipeline. Run with: pytest -q

These avoid LLM calls (reply.py has a deterministic fallback) and use tiny inputs
so they finish fast on CI.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Force a temp DB so tests don't touch any local state.
_TMP = Path(tempfile.mkdtemp(prefix="ai_support_test_"))
os.environ["DB_URL"] = f"sqlite:///{_TMP / 'tickets.db'}"
os.environ["CHROMA_DIR"] = str(_TMP / "chroma")
os.environ["GROQ_API_KEY"] = ""  # force fallback replies

from app.data.synthesize import generate_dataset  # noqa: E402
from app.pipeline.classify import classify  # noqa: E402
from app.pipeline.clean import clean_text, is_valid  # noqa: E402
from app.pipeline.reply import suggest_reply  # noqa: E402
from app.pipeline.runner import process_csv  # noqa: E402
from app.pipeline.sentiment import analyze  # noqa: E402


def test_clean_basics():
    assert clean_text("Hello\xa0there!  ") == "Hello there!"
    assert clean_text(None) == ""
    assert clean_text("https://x.com and a@b.com #123456") == "[URL] and [EMAIL] [ORDER]"
    assert is_valid("hi there")
    assert not is_valid("")
    assert not is_valid("  ")
    assert not is_valid("hi")  # below 3-char minimum


def test_sentiment_negative_is_frustrated():
    out = analyze("This is the WORST experience EVER!!! I am furious and demand a refund now.")
    assert out["sentiment"] == "negative"
    assert out["frustration_level"] >= 3


def test_sentiment_positive():
    out = analyze("Absolutely love the product, fantastic experience, will buy again.")
    assert out["sentiment"] == "positive"
    assert out["frustration_level"] <= 1


def test_classifier_routes_obvious_cases():
    cat, _ = classify("My package is delayed and tracking hasn't updated in a week.")
    assert cat == "Shipping & Delivery"
    cat, _ = classify("I was double charged on my credit card for order #123456.")
    assert cat == "Payment & Billing"
    cat, _ = classify("The product arrived broken and the battery is dead.")
    assert cat == "Product Defect / Quality"


def test_reply_fallback_when_no_api_key():
    reply = suggest_reply("My package never arrived.", category="Shipping & Delivery")
    assert isinstance(reply, str)
    assert len(reply) > 20


def test_end_to_end_pipeline(tmp_path):
    csv_path = tmp_path / "mini.csv"
    generate_dataset(csv_path, n=120, seed=7)
    summary = process_csv(str(csv_path))
    assert summary["received"] == 120
    assert summary["processed"] >= 110  # allow a few dropped if any cleaning kicks in
    assert summary["skipped"] + summary["processed"] == summary["received"]
