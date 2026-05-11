"""Lightweight text cleaning for inbound support messages."""
from __future__ import annotations

import html
import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_ORDER_RE = re.compile(r"#\s?\d{4,}")
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = _CTRL_RE.sub(" ", text)
    text = _URL_RE.sub("[URL]", text)
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _ORDER_RE.sub("[ORDER]", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def is_valid(text: str | None, min_chars: int = 3) -> bool:
    if not text:
        return False
    return len(text.strip()) >= min_chars
