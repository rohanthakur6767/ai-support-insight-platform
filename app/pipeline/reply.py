"""Suggested-reply generator using Groq's free Llama 3.3 70B endpoint.

Falls back to a deterministic, category-specific template if the API key is
missing or the call fails, so the rest of the pipeline still works offline.
"""
from __future__ import annotations

import logging
import textwrap
from functools import lru_cache

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)


FALLBACK_TEMPLATES: dict[str, str] = {
    "Shipping & Delivery": (
        "Hi there, thanks for reaching out and I'm sorry your order is delayed. "
        "I'm escalating this to our carrier team now and will follow up within "
        "24 hours with a tracking update. If we can't confirm delivery by then, "
        "I'll arrange a reship at no cost."
    ),
    "Returns & Refunds": (
        "Hi there, no problem — I'd be happy to help. I've started the return "
        "and emailed you a prepaid label. Once the item is received the refund "
        "will post to your original payment method within 5–7 business days."
    ),
    "Payment & Billing": (
        "Hi there, apologies for the billing trouble. I can see the issue on "
        "your account and have reversed the charge. Please allow 3–5 business "
        "days for the credit to appear on your statement."
    ),
    "Product Defect / Quality": (
        "Hi there, I'm really sorry the product isn't working as expected — "
        "this isn't the experience we want for you. I've arranged a free "
        "replacement and emailed a prepaid return label for the defective unit."
    ),
    "Order Status": (
        "Hi there, your order is currently in our fulfilment centre and is "
        "expected to ship within 24 hours. You'll receive tracking details by "
        "email as soon as it leaves the warehouse."
    ),
    "Account & Login": (
        "Hi there, I've unlocked your account and sent a fresh password reset "
        "link. If it doesn't arrive within 10 minutes please check your spam "
        "folder, or let me know and I'll switch the email on file."
    ),
    "Promotions & Discounts": (
        "Hi there, sorry that promo didn't apply at checkout. As a one-time "
        "courtesy I've manually credited the discount to your order — you "
        "should see the adjustment within a few hours."
    ),
    "Cancellation": (
        "Hi there, I've cancelled the order and a full refund will be issued "
        "within 3–5 business days. Sorry it didn't work out this time."
    ),
    "Other": (
        "Hi there, thanks for getting in touch! Happy to help — could you "
        "share a few more details so I can point you to the right resource?"
    ),
}


SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an experienced e-commerce customer-support agent. Write a single,
    short reply (max 90 words) to the customer message provided.

    Rules:
    - Open with a brief empathetic acknowledgement.
    - State the concrete next step you (the agent) will take.
    - Give a realistic ETA or expectation.
    - Be warm but professional. No emojis. No marketing copy.
    - Do not invent order numbers, names, or refund amounts that aren't in the message.
    - Reply in the customer's language if it's clearly non-English; otherwise English.
    Output only the reply text, nothing else.
    """
).strip()


@lru_cache(maxsize=1)
def _client():
    cfg = get_settings()
    if not cfg.groq_api_key:
        return None
    try:
        from groq import Groq
    except ImportError:
        logger.warning("groq package not installed; using fallback replies")
        return None
    return Groq(api_key=cfg.groq_api_key)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=4))
def _call_groq(message: str, category: str | None, product: str | None) -> str:
    client = _client()
    if client is None:
        raise RuntimeError("No Groq client configured")
    cfg = get_settings()
    ctx_lines = []
    if category:
        ctx_lines.append(f"Detected category: {category}")
    if product:
        ctx_lines.append(f"Product: {product}")
    context = ("\n".join(ctx_lines) + "\n\n") if ctx_lines else ""

    chat = client.chat.completions.create(
        model=cfg.groq_model,
        temperature=0.4,
        max_tokens=220,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}Customer message:\n{message}"},
        ],
    )
    return chat.choices[0].message.content.strip()


def suggest_reply(
    message: str,
    category: str | None = None,
    product: str | None = None,
) -> str:
    if not message:
        return ""
    try:
        if _client() is not None:
            return _call_groq(message, category, product)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groq reply generation failed (%s); using fallback", exc)
    return FALLBACK_TEMPLATES.get(category or "Other", FALLBACK_TEMPLATES["Other"])
