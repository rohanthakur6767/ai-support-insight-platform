# Business Thinking

## 1. Top 3 insights for leadership

1. **Revenue at risk by issue category, this week vs last.**
   A single dollar figure that says: "open + frustrated tickets touching $X of
   orders, up Y% w/w." This is the headline a VP can act on — it tells them
   whether the support backlog is growing in *value*, not just in *count*. The
   `/insights/summary` endpoint surfaces this and the dashboard renders it as
   the first KPI on the overview page.

2. **Top recurring issues and which products they cluster on.**
   Not "10% of tickets are shipping issues" (too vague) but "Cluster #3 — '
   *package never arrived after 14 days*' — 412 tickets, 78% touching Wireless
   Headphones, avg order $145." That is operationally actionable: it points
   the ops team at a specific carrier-lane / SKU combination.

3. **Sentiment trajectory.**
   A daily stacked area of positive / neutral / negative tickets. Leadership
   doesn't care about today's number — they care about whether negative volume
   is trending up over the last 14 days. Combined with the issue clusters this
   tells them *why*.

## 2. How this reduces support costs

- **Suggested replies** cut average handle time. If an agent accepts the draft
  reply unchanged 30% of the time and edits it the rest, an industry-typical
  4-minute reply shrinks to ~90 seconds — a >50% reduction on draft cost.
- **Frustration routing**. Frustration ≥ 3 should auto-escalate to a senior
  agent. This avoids the well-documented multi-touch tax where a junior agent
  closes a ticket the customer then re-opens, doubling cost-per-resolution.
- **Self-service deflection.** The clustered "top issues" view feeds product
  and marketing: the team can ship a help-centre article or fix a checkout bug
  that eliminates the *source* of a cluster, not just its symptoms. A single
  cluster of 400+ tickets/month deflected at $5/ticket = ~$24k/year saved.
- **Knowledge concentration.** Vector search ("show me past tickets like this
  one") removes the long tail of "let me ask my manager" pauses for new
  agents — a 10–20% productivity lift in the first 90 days of tenure.

## 3. How this increases revenue / retention

- **Saving high-value frustrated customers.** The `revenue_at_risk` KPI
  isolates the subset of frustrated tickets tied to *large orders*. A
  prioritised outreach (callback, expedited refund, credit) on the top 5% of
  these recoups customers who would otherwise churn. Industry benchmark: 5%
  retention lift → ~25%+ CLV uplift.
- **Repeat-purchase signal.** Mapping cluster → product surfaces *which SKUs*
  generate the most negative tickets. Pulling those SKUs (or fixing them with
  the supplier) cuts return rates, raises NPS, and increases repeat order
  rate.
- **Cross-channel parity.** Sentiment broken out by channel highlights which
  channel (chat / email / web) is silently degrading. Investing there raises
  conversion from support interaction to next purchase.

## 4. Metrics the company should track

| Metric                              | Why it matters                                                              | Target direction |
| ----------------------------------- | --------------------------------------------------------------------------- | ---------------- |
| First-Response Time (FRT)           | Faster acknowledgement → less frustration & churn                            | ↓ |
| Average Handle Time (AHT)           | Direct cost driver; suggested-reply adoption should lower it                 | ↓ |
| First-Contact Resolution (FCR)      | Multi-touch tickets cost 2–3× a one-touch ticket                             | ↑ |
| Frustration ≥ 3 share               | Early warning for churn risk; should drop as causes get fixed                | ↓ |
| Revenue-at-risk (rolling 7d)        | Translates support load into a CFO-readable number                           | ↓ |
| Top-issue concentration             | High % concentrated in 3 clusters = quick wins available                     | Watch |
| Sentiment compound 14-day MA        | Smoothed read on customer mood                                               | ↑ |
| Cluster lifetime                    | How fast new clusters get fixed (engineering / supplier responsiveness)      | ↓ |
| Suggested-reply acceptance rate     | Adoption signal for the AI feature itself                                    | ↑ |
| Cost per resolved ticket            | The ultimate efficiency metric — should drop with all of the above           | ↓ |
