"""Embedding model singleton + Chroma vector-store helper.

Uses a small sentence-transformer model so the pipeline runs free, locally, and fast.
The same embeddings power: zero-shot classification, KMeans issue clustering, and the
agent-facing semantic-search lookup served by /tickets/search.
"""
from __future__ import annotations

import functools
import logging
from typing import Iterable, Sequence

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    cfg = get_settings()
    logger.info("Loading embedding model: %s", cfg.embed_model)
    return SentenceTransformer(cfg.embed_model)


def encode(texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    model = get_embedder()
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


@functools.lru_cache(maxsize=1)
def get_chroma():
    cfg = get_settings()
    client = chromadb.PersistentClient(
        path=cfg.chroma_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name="tickets", metadata={"hnsw:space": "cosine"})


def upsert_vectors(
    ticket_ids: Iterable[str],
    texts: Iterable[str],
    metadatas: Iterable[dict],
) -> None:
    ids = list(ticket_ids)
    if not ids:
        return
    docs = list(texts)
    metas = list(metadatas)
    embeds = encode(docs).tolist()
    col = get_chroma()
    col.upsert(ids=ids, embeddings=embeds, documents=docs, metadatas=metas)


def search(query: str, k: int = 5) -> list[dict]:
    col = get_chroma()
    q = encode([query]).tolist()
    res = col.query(query_embeddings=q, n_results=k)
    out = []
    for i, doc in enumerate(res["documents"][0]):
        out.append(
            {
                "ticket_id": res["ids"][0][i],
                "document": doc,
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i] if "distances" in res else None,
            }
        )
    return out
