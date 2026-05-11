"""Sentiment and frustration scoring.

VADER handles the 'is this positive / negative' axis; a small set of heuristic
features (caps, exclamation, repeat-contact phrasing, profanity-lite) handle the
'how frustrated is this customer' axis. Frustration is a 0–4 ordinal, where
3+ should escalate to a senior agent.
"""
from __future__ import annotations

import functools
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_FRUSTRATION_CUES = [
    r"\b(third|fourth|fifth) time\b",
    r"unacceptable",
    r"worst",
    r"terrible",
    r"furious",
    r"disgusted",
    r"appall(ed|ing)",
    r"never again",
    r"switch(ing)? to (a )?competitor",
    r"dispute (the )?charge",
    r"file a complaint",
    r"lawsuit|sue\b|attorney",
    r"refund (now|immediately|today)",
    r"cancel (my|the) (order|account|subscription)",
]
_FRUSTRATION_RE = re.compile("|".join(_FRUSTRATION_CUES), re.IGNORECASE)


@functools.lru_cache(maxsize=1)
def _vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def sentiment(text: str) -> tuple[str, float]:
    scores = _vader().polarity_scores(text or "")
    compound = scores["compound"]
    if compound >= 0.25:
        label = "positive"
    elif compound <= -0.25:
        label = "negative"
    else:
        label = "neutral"
    return label, compound


def frustration(text: str, sentiment_score: float | None = None) -> int:
    if not text:
        return 0
    level = 0
    if sentiment_score is None:
        sentiment_score = _vader().polarity_scores(text)["compound"]

    if sentiment_score <= -0.6:
        level += 2
    elif sentiment_score <= -0.25:
        level += 1

    if _FRUSTRATION_RE.search(text):
        level += 1

    excl = text.count("!")
    if excl >= 3:
        level += 1
    elif excl >= 1:
        level += 0  # one exclamation is normal

    # ALL-CAPS words longer than 3 chars (RAGE shouting)
    caps_words = sum(1 for w in text.split() if len(w) > 3 and w.isupper())
    if caps_words >= 2:
        level += 1

    return max(0, min(4, level))


def analyze(text: str) -> dict:
    label, score = sentiment(text)
    return {
        "sentiment": label,
        "sentiment_score": round(score, 4),
        "frustration_level": frustration(text, score),
    }
