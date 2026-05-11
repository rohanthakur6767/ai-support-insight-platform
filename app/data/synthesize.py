"""Synthetic customer-support ticket generator.

Generates a realistic, reproducible CSV of ~5k tickets across 9 problem categories
with controllable distributions, sentiment variety, and a manufactured spike that the
analytics layer can later surface (without us needing a paid API).

The data is template-driven rather than LLM-driven so the generator is:
  - free to run
  - reproducible from a seed
  - fast (whole 5k set in <2s)
"""
from __future__ import annotations

import csv
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

PRODUCTS = [
    "Wireless Headphones",
    "Smart Watch",
    "Running Shoes",
    "Coffee Maker",
    "Yoga Mat",
    "Bluetooth Speaker",
    "Backpack",
    "Sunglasses",
    "Mechanical Keyboard",
    "Air Purifier",
]

COUNTRIES = ["US", "UK", "DE", "FR", "IN", "CA", "AU", "JP", "BR", "MX"]
CHANNELS = ["chat", "email", "web"]
STATUSES = ["open", "in_progress", "resolved", "escalated"]

# Per-category message templates. Each template has slots for product/order details.
TEMPLATES: dict[str, list[str]] = {
    "Shipping & Delivery": [
        "My order for {product} was supposed to arrive {days} days ago but it's still not here. Tracking hasn't updated in a week.",
        "The {product} I ordered shows as delivered but I never received it. Can you investigate?",
        "Where is my package? Order #{order_id} was placed {days} days ago and there's no shipping update.",
        "Shipping has been delayed three times now. This is unacceptable — I need the {product} by Friday.",
        "Hi, can I get an ETA on order #{order_id}? The {product} was supposed to ship last week.",
        "Package arrived damaged. The {product} box was crushed and the item inside is broken.",
        "Wrong address on my shipment for the {product}. How do I redirect it?",
    ],
    "Returns & Refunds": [
        "I'd like to return the {product} I bought last week. It doesn't fit my needs.",
        "I returned the {product} two weeks ago and haven't seen the refund yet. Order #{order_id}.",
        "How do I start a return? The {product} arrived but I changed my mind.",
        "Your return policy says 30 days but the portal won't let me submit. Please help!",
        "Refund for order #{order_id} is showing as processed but nothing has hit my card.",
        "I want a full refund — the {product} is nothing like the description.",
    ],
    "Payment & Billing": [
        "I was charged twice for the same {product}. Please refund the duplicate charge on order #{order_id}.",
        "My credit card keeps getting declined at checkout but the bank confirms it's fine.",
        "There's a mystery charge of ${amt} on my statement. I never ordered this.",
        "Coupon code didn't apply at checkout and I was charged full price for the {product}.",
        "Invoice for order #{order_id} shows the wrong tax amount. Can you fix?",
        "I'm being billed for a subscription I cancelled three months ago.",
    ],
    "Product Defect / Quality": [
        "The {product} stopped working after just {days} days. Total junk.",
        "Quality of the {product} is much worse than advertised. Cheap materials.",
        "Battery on my {product} drains in two hours. Clearly defective.",
        "The {product} has a manufacturing defect — there's a crack right out of the box.",
        "My {product} overheats whenever I use it. This feels like a safety issue.",
        "Sound quality on the {product} is terrible. Want an exchange.",
    ],
    "Order Status": [
        "Can you confirm order #{order_id} went through? I never got a confirmation email.",
        "What's the status of my {product} order? It still shows 'processing' after {days} days.",
        "When will my order ship? I placed it on {date_str}.",
        "Order #{order_id} — is this still on track for delivery this week?",
    ],
    "Account & Login": [
        "I can't log in to my account. Password reset link never arrives.",
        "My account is locked. I tried logging in a couple of times and now it says suspended.",
        "I changed my email but now I can't access my order history.",
        "Two-factor isn't working — I'm not getting the SMS code.",
        "Please delete my account and all associated data under GDPR.",
    ],
    "Promotions & Discounts": [
        "The 20% off code on your homepage isn't working on the {product}.",
        "I missed the Black Friday sale by an hour. Any chance you can honor the price?",
        "Why doesn't the loyalty discount stack with the seasonal promo?",
        "Promo code SAVE15 says expired but your email said it runs until next week.",
    ],
    "Cancellation": [
        "Please cancel order #{order_id} immediately. I no longer need the {product}.",
        "I tried to cancel within the 1-hour window and the system wouldn't let me.",
        "Cancel my subscription effective today. No further charges please.",
        "Need to cancel — ordered the wrong size of the {product}.",
    ],
    "Other": [
        "Just wanted to say the {product} is fantastic. Best purchase I've made all year!",
        "Do you ship to {country}? Couldn't find that info on the site.",
        "Is the {product} compatible with my older model?",
        "Curious whether you have a B2B / wholesale program.",
        "Hi, the website checkout page is throwing a JavaScript error in Firefox.",
    ],
}

# Approximate share each category should take in the final dataset.
CATEGORY_WEIGHTS = {
    "Shipping & Delivery": 0.24,
    "Returns & Refunds": 0.16,
    "Payment & Billing": 0.13,
    "Product Defect / Quality": 0.14,
    "Order Status": 0.11,
    "Account & Login": 0.07,
    "Promotions & Discounts": 0.05,
    "Cancellation": 0.05,
    "Other": 0.05,
}

# Frustration boosters appended to a fraction of messages.
FRUSTRATION_SUFFIXES = [
    " This is the third time I'm contacting you about this.",
    " I'm extremely disappointed.",
    " If this isn't fixed today I'm disputing the charge.",
    " Honestly considering switching to a competitor.",
    " Absolute worst customer experience ever.",
]

# Agent reply templates per category (used as ground truth for evaluating suggested-reply).
AGENT_REPLIES = {
    "Shipping & Delivery": "Hi {name}, thanks for reaching out — I'm sorry your order is delayed. I'm checking with our carrier on tracking #{order_id} and will follow up within 24 hours with an update. If we can't confirm delivery by then, I'll process a reship at no charge.",
    "Returns & Refunds": "Hi {name}, no problem at all. I've initiated the return for order #{order_id} and emailed you a prepaid label. Once we receive the item the refund will post within 5–7 business days.",
    "Payment & Billing": "Hi {name}, apologies for the billing confusion. I can see the duplicate charge on order #{order_id} and have refunded it — please allow 3–5 business days for it to appear on your statement.",
    "Product Defect / Quality": "Hi {name}, I'm sorry the {product} isn't performing as expected. This isn't the experience we want for you — I've arranged a free replacement and a return label is on the way. Please discard the damaged unit safely.",
    "Order Status": "Hi {name}, your order #{order_id} is currently in our fulfillment center and is expected to ship within 24 hours. You'll get tracking by email as soon as it leaves the warehouse.",
    "Account & Login": "Hi {name}, I've unlocked your account and sent a new password reset link. If the email doesn't arrive in 10 minutes please check spam or let me know and I'll switch the address.",
    "Promotions & Discounts": "Hi {name}, sorry the code didn't apply — that promotion has some category exclusions. I've manually credited the 20% off to your order as a one-time courtesy.",
    "Cancellation": "Hi {name}, I've cancelled order #{order_id} and a full refund will be issued within 3–5 business days. Sorry it didn't work out this time.",
    "Other": "Hi {name}, thanks for getting in touch! Happy to help — could you share a few more details so I can point you to the right resource?",
}


def _resolve_status(category: str, frustration: int) -> str:
    if category == "Other":
        return random.choices(["resolved", "open"], weights=[0.8, 0.2])[0]
    if frustration >= 3:
        return random.choices(["escalated", "in_progress", "resolved"], weights=[0.5, 0.3, 0.2])[0]
    return random.choices(STATUSES, weights=[0.25, 0.20, 0.50, 0.05])[0]


def _frustration_level(category: str, message: str) -> int:
    base = {
        "Shipping & Delivery": 2,
        "Returns & Refunds": 2,
        "Payment & Billing": 3,
        "Product Defect / Quality": 3,
        "Order Status": 1,
        "Account & Login": 2,
        "Promotions & Discounts": 1,
        "Cancellation": 2,
        "Other": 0,
    }[category]
    if any(suffix.strip() in message for suffix in FRUSTRATION_SUFFIXES):
        base += 1
    if "!" in message or message.isupper():
        base += 1
    return max(0, min(4, base + random.choice([-1, 0, 0, 0, 1])))


def _build_message(category: str, product: str, order_id: str, days: int, when: datetime, country: str) -> str:
    tpl = random.choice(TEMPLATES[category])
    msg = tpl.format(
        product=product,
        order_id=order_id,
        days=days,
        amt=round(random.uniform(15, 350), 2),
        date_str=when.strftime("%b %d"),
        country=country,
    )
    if random.random() < 0.20:
        msg += random.choice(FRUSTRATION_SUFFIXES)
    return msg


def _iter_tickets(n: int, start: datetime, end: datetime, spike_day: datetime | None) -> Iterator[dict]:
    cats = list(CATEGORY_WEIGHTS.keys())
    weights = list(CATEGORY_WEIGHTS.values())
    span = (end - start).total_seconds()

    for i in range(n):
        # Time: most tickets distributed evenly; a small fraction concentrated on spike_day.
        if spike_day and random.random() < 0.06:
            ts = spike_day + timedelta(seconds=random.randint(0, 86_400))
            category = "Shipping & Delivery"  # spike represents a shipping incident
        else:
            ts = start + timedelta(seconds=random.uniform(0, span))
            category = random.choices(cats, weights=weights)[0]

        product = random.choice(PRODUCTS)
        country = random.choice(COUNTRIES)
        order_id = str(random.randint(100_000, 999_999))
        days = random.randint(2, 21)
        message = _build_message(category, product, order_id, days, ts, country)
        frustration = _frustration_level(category, message)

        order_value = round(random.uniform(15, 500), 2)
        # Inflate revenue exposure for defect / shipping issues so leadership-style insights bite.
        if category in ("Product Defect / Quality", "Shipping & Delivery"):
            order_value *= random.uniform(1.0, 1.6)

        reply = AGENT_REPLIES[category].format(
            name=f"customer-{random.randint(1, 9999)}",
            order_id=order_id,
            product=product,
        )

        yield {
            "ticket_id": f"TCK-{uuid.uuid4().hex[:10].upper()}",
            "timestamp": ts.isoformat(timespec="seconds"),
            "customer_id": f"CUST-{random.randint(1000, 99999)}",
            "channel": random.choice(CHANNELS),
            "message": message,
            "agent_reply": reply if random.random() < 0.7 else "",
            "product": product,
            "order_value": round(order_value, 2),
            "customer_country": country,
            "resolution_status": _resolve_status(category, frustration),
            # ground-truth fields (not used by the pipeline at inference time, but handy for evaluation)
            "true_category": category,
            "true_frustration": frustration,
        }


def generate_dataset(
    path: str | Path,
    n: int = 5000,
    seed: int = 42,
    days_back: int = 60,
    inject_spike: bool = True,
) -> Path:
    random.seed(seed)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    end = datetime.utcnow().replace(microsecond=0)
    start = end - timedelta(days=days_back)
    spike_day = end - timedelta(days=10) if inject_spike else None

    fieldnames = [
        "ticket_id",
        "timestamp",
        "customer_id",
        "channel",
        "message",
        "agent_reply",
        "product",
        "order_value",
        "customer_country",
        "resolution_status",
        "true_category",
        "true_frustration",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in _iter_tickets(n, start, end, spike_day):
            writer.writerow(row)

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic support tickets")
    parser.add_argument("--out", default="data/tickets.csv")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    p = generate_dataset(args.out, n=args.n, seed=args.seed)
    print(f"Wrote {args.n} tickets to {p}")
