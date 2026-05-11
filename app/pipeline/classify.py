"""Zero-shot ticket classifier.

For each category we hand-author a short *prototype* description. At classification
time we embed the ticket and the prototypes, then assign the ticket to the
nearest-cosine prototype. This avoids needing labelled training data or an LLM
call per ticket, while still being trivial to extend (just add a new category +
prototype).
"""
from __future__ import annotations

import functools

import numpy as np

from app.config import get_settings
from app.pipeline.embed import encode

# Each prototype is a concatenation of phrases a customer might actually use.
CATEGORY_PROTOTYPES: dict[str, str] = {
    "Shipping & Delivery": (
        "package delayed, tracking not updating, where is my order, "
        "delivery never arrived, carrier issue, shipment damaged in transit"
    ),
    "Returns & Refunds": (
        "I want to return this item, refund not received, return label, "
        "money back, restocking fee, exchange for a different size"
    ),
    "Payment & Billing": (
        "double charged, credit card declined, wrong amount on invoice, "
        "billing dispute, mystery charge on statement, tax incorrect"
    ),
    "Product Defect / Quality": (
        "item broken, stopped working, defective, poor quality, "
        "battery drains fast, manufacturing defect, item arrived broken"
    ),
    "Order Status": (
        "is my order placed, order confirmation not received, "
        "still processing after days, when will it ship, ETA"
    ),
    "Account & Login": (
        "cannot log in, password reset not working, account locked, "
        "two factor authentication, delete my account, change email"
    ),
    "Promotions & Discounts": (
        "promo code not working, coupon expired, missed sale price, "
        "discount not applying, loyalty points"
    ),
    "Cancellation": (
        "please cancel my order, cancel subscription, "
        "want to cancel before it ships, cancellation window"
    ),
    "Other": (
        "general question, product compatibility, shipping policy question, "
        "feedback, website bug, compliment"
    ),
}


@functools.lru_cache(maxsize=1)
def _prototype_matrix() -> tuple[list[str], np.ndarray]:
    cats = get_settings().categories
    descs = [CATEGORY_PROTOTYPES[c] for c in cats]
    matrix = encode(descs)
    return cats, matrix


def classify(text: str) -> tuple[str, float]:
    cats, matrix = _prototype_matrix()
    vec = encode([text])[0]
    sims = matrix @ vec  # cosine, since both are normalized
    idx = int(np.argmax(sims))
    return cats[idx], float(sims[idx])


def classify_batch(texts: list[str]) -> list[tuple[str, float]]:
    cats, matrix = _prototype_matrix()
    if not texts:
        return []
    vecs = encode(texts)
    sims = vecs @ matrix.T  # (n, k)
    idxs = sims.argmax(axis=1)
    return [(cats[i], float(sims[r, i])) for r, i in enumerate(idxs)]
