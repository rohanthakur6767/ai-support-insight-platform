"""Cluster recent tickets to surface the top recurring issues.

Approach: KMeans on the message embeddings, then for each cluster pick the
ticket closest to the centroid as a representative, derive a short label from
that representative's leading clause, and aggregate metrics (count, average
order value, share of frustrated customers) per cluster.

Why KMeans vs HDBSCAN: with ~5k tickets and a high category-overlap distribution,
KMeans gives stable cluster counts that are easy to surface in a dashboard. We
keep K modest (default 12) to bias toward broad, actionable issue groups instead
of long-tail noise.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from app.pipeline.embed import encode

logger = logging.getLogger(__name__)


@dataclass
class IssueCluster:
    cluster_id: int
    label: str
    count: int
    avg_order_value: float
    avg_frustration: float
    sample_messages: list[str]
    member_ticket_ids: list[str]


_FILLER_RE = re.compile(r"^(hi|hello|hey|please|thanks?|thank you|sorry)[,!.\s]+", re.IGNORECASE)


def _shorten(msg: str, max_chars: int = 80) -> str:
    msg = _FILLER_RE.sub("", msg.strip())
    # cut at first sentence-ending punctuation
    for sep in (". ", "? ", "! "):
        i = msg.find(sep)
        if 15 < i < max_chars:
            return msg[:i].strip().rstrip(",.!?")
    return msg[:max_chars].strip().rstrip(",.!?")


def cluster_issues(
    ticket_ids: list[str],
    messages: list[str],
    order_values: list[float],
    frustration_levels: list[int],
    k: int = 12,
    random_state: int = 42,
) -> list[IssueCluster]:
    if not messages:
        return []
    from sklearn.cluster import KMeans  # heavy — import on demand

    k = min(k, max(1, len(messages) // 4))  # keep clusters sized > a handful
    vecs = encode(messages)
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(vecs)

    clusters: list[IssueCluster] = []
    for cid in range(k):
        member_mask = labels == cid
        if not member_mask.any():
            continue
        member_idx = np.where(member_mask)[0]
        # representative = closest to centroid
        centroid = km.cluster_centers_[cid]
        dists = np.linalg.norm(vecs[member_idx] - centroid, axis=1)
        rep_idx = int(member_idx[int(np.argmin(dists))])
        sample = [messages[i] for i in member_idx[np.argsort(dists)[:3]]]

        clusters.append(
            IssueCluster(
                cluster_id=int(cid),
                label=_shorten(messages[rep_idx]),
                count=int(member_mask.sum()),
                avg_order_value=float(np.mean([order_values[i] for i in member_idx])),
                avg_frustration=float(np.mean([frustration_levels[i] for i in member_idx])),
                sample_messages=sample,
                member_ticket_ids=[ticket_ids[i] for i in member_idx],
            )
        )

    clusters.sort(key=lambda c: c.count, reverse=True)
    return clusters
