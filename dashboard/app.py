"""Streamlit dashboard for the AI Support Insight Platform.

Reads everything via the FastAPI backend (no DB coupling here). Set the
backend URL via the API_BASE_URL env var if it isn't the default localhost:8000.
"""
from __future__ import annotations

import os
from datetime import datetime

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Support Insights",
    page_icon="📞",
    layout="wide",
)


# ---- helpers --------------------------------------------------------------

@st.cache_data(ttl=60)
def _get(path: str, **params):
    r = httpx.get(f"{API_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path: str, json=None, files=None):
    r = httpx.post(f"{API_BASE}{path}", json=json, files=files, timeout=120)
    r.raise_for_status()
    return r.json()


def _badge(label: str, value, help_text: str = ""):
    st.metric(label, value, help=help_text)


# ---- sidebar --------------------------------------------------------------

st.sidebar.title("🛟 Support Insights")
st.sidebar.caption(f"API: `{API_BASE}`")

try:
    health = _get("/health")
    st.sidebar.success(f"Backend healthy ({health.get('status')})")
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Backend unreachable: {exc}")
    st.stop()

days = st.sidebar.slider("Window (days)", 7, 90, 30)

page = st.sidebar.radio(
    "View",
    ["Overview", "Top Issues", "Tickets", "Agent Assist", "Upload"],
    index=0,
)


# ---- pages ----------------------------------------------------------------

def page_overview():
    st.title("Overview")
    st.caption(f"Rolling {days}-day window")

    data = _get("/insights/summary", days=days)
    total = data["total_tickets"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _badge("Tickets", f"{total:,}")
    with c2:
        _badge("Avg frustration", f"{data['avg_frustration']:.2f}", "Scale 0–4")
    with c3:
        _badge("Revenue at risk", f"${data['revenue_at_risk']:,.0f}",
               "Order value tied to unresolved + frustrated tickets")
    with c4:
        neg = sum(p["negative"] for p in data["sentiment_trend"])
        _badge("Negative msgs", f"{neg:,}")

    if total == 0:
        st.info("No tickets in this window yet. Try the Upload tab or run the seed script.")
        return

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Sentiment trend")
        if data["sentiment_trend"]:
            df = pd.DataFrame(data["sentiment_trend"])
            df_melt = df.melt(id_vars="date", var_name="sentiment", value_name="count")
            fig = px.area(
                df_melt,
                x="date",
                y="count",
                color="sentiment",
                color_discrete_map={"positive": "#22c55e", "neutral": "#94a3b8", "negative": "#ef4444"},
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend_title=None, height=360)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Category mix")
        if data["by_category"]:
            df = pd.DataFrame(data["by_category"])
            fig = px.bar(df, x="count", y="category", orientation="h", text="pct")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=360, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Revenue exposure by category")
    rev = _get("/insights/revenue-by-category", days=days)
    if rev:
        df = pd.DataFrame(rev)
        fig = px.bar(
            df,
            x="category",
            y="revenue_touched",
            color="avg_frustration",
            color_continuous_scale="Reds",
            hover_data=["tickets"],
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=340)
        st.plotly_chart(fig, use_container_width=True)


def page_top_issues():
    st.title("Top recurring issues")
    st.caption("KMeans clusters over message embeddings, ranked by volume.")
    data = _get("/insights/summary", days=days)
    if not data["top_issues"]:
        st.info("Top issues will appear once the pipeline has clustered enough tickets (>24).")
        return

    for issue in data["top_issues"]:
        with st.expander(f"#{issue['cluster_id']} — {issue['label']}  ({issue['count']} tickets)"):
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Volume", issue["count"])
            with c2:
                st.metric("Avg order value", f"${issue['avg_order_value']:.0f}")
            st.markdown("**Representative messages**")
            for s in issue["sample_messages"]:
                st.markdown(f"> {s}")


def page_tickets():
    st.title("Tickets")
    cols = st.columns(4)
    category = cols[0].selectbox(
        "Category",
        ["All"] + [
            "Shipping & Delivery", "Returns & Refunds", "Payment & Billing",
            "Product Defect / Quality", "Order Status", "Account & Login",
            "Promotions & Discounts", "Cancellation", "Other",
        ],
    )
    sentiment = cols[1].selectbox("Sentiment", ["All", "positive", "neutral", "negative"])
    min_frust = cols[2].slider("Min frustration", 0, 4, 0)
    limit = cols[3].number_input("Limit", min_value=10, max_value=500, value=50, step=10)

    params = {"limit": int(limit)}
    if category != "All":
        params["category"] = category
    if sentiment != "All":
        params["sentiment"] = sentiment
    if min_frust > 0:
        params["min_frustration"] = min_frust

    rows = _get("/tickets", **params)
    if not rows:
        st.info("No tickets match these filters.")
        return

    df = pd.DataFrame(rows)
    show_cols = [
        "ticket_id", "timestamp", "channel", "category", "sentiment",
        "frustration_level", "order_value", "resolution_status", "message",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Ticket detail")
    tid = st.selectbox("Pick a ticket", df["ticket_id"].tolist())
    if tid:
        ticket = _get(f"/tickets/{tid}")
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Customer message**")
            st.write(ticket["message"])
            st.markdown("**Logged agent reply**")
            st.write(ticket.get("agent_reply") or "_(none recorded)_")
        with col_r:
            for k in ("category", "sentiment", "frustration_level", "resolution_status",
                      "order_value", "customer_country", "channel"):
                if ticket.get(k) is not None:
                    st.write(f"**{k}:** {ticket[k]}")


def page_agent_assist():
    st.title("Agent assist")
    st.caption("Paste an incoming message — get a suggested reply, detected category, and a frustration read.")
    msg = st.text_area("Customer message", height=150, placeholder="Hi, my order arrived broken...")
    product = st.text_input("Product (optional)")
    if st.button("Generate suggestion", type="primary", disabled=not msg.strip()):
        with st.spinner("Calling LLM..."):
            res = _post("/tickets/reply", json={"message": msg, "product": product or None})
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", res["category"])
        c2.metric("Sentiment", res["sentiment"])
        c3.metric("Frustration", res["frustration_level"])
        st.markdown("### Suggested reply")
        st.success(res["suggested_reply"])


def page_upload():
    st.title("Upload tickets")
    st.caption("Drop a CSV with at least a `message` column. Other fields are optional.")
    f = st.file_uploader("CSV file", type=["csv"])
    if f is not None and st.button("Process", type="primary"):
        with st.spinner("Running pipeline..."):
            res = _post("/tickets/upload", files={"file": (f.name, f.getvalue(), "text/csv")})
        st.success(f"Received {res['received']}, processed {res['processed']}, skipped {res['skipped']}")
        st.cache_data.clear()


PAGES = {
    "Overview": page_overview,
    "Top Issues": page_top_issues,
    "Tickets": page_tickets,
    "Agent Assist": page_agent_assist,
    "Upload": page_upload,
}
PAGES[page]()
