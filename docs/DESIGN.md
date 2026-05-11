# Design Document

## 1. AI Choices

| Stage              | Approach                                            | Why this, not the alternative                                                                                                  |
| ------------------ | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Cleaning           | Regex normalisation (URLs / emails / order numbers → tokens) | Tickets contain noisy PII-ish content. Token replacement keeps the signal while protecting downstream embeddings from cardinality blow-up. |
| Embeddings         | `all-MiniLM-L6-v2` (384-d) from sentence-transformers | Free, runs on CPU, ~80 MB. Quality is sufficient for retail-support phrasing; using OpenAI/Voyage embeddings would 10× the cost without moving accuracy materially. |
| Categorisation     | Zero-shot embedding similarity to category prototypes | We control the taxonomy and don't have labelled training data. Zero-shot is trivially extensible (add a new category = add a sentence) and avoids per-ticket LLM cost. |
| Sentiment          | VADER + heuristic boosters                          | Lexicon-based sentiment is good enough for support copy and runs in microseconds. A finetuned transformer would buy maybe 3–5% accuracy at 100× the latency. |
| Frustration        | Sentiment compound + linguistic cues (caps, "third time", legalese) | Two-axis read: a customer can be *negative* (defective product) without being *frustrated* (calm tone). Frustration drives routing — that's why it's separate. |
| Top issues         | KMeans (k=12) on embeddings, label from cluster-centroid representative | Tickets sit on a manifold dominated by ~10 topics. KMeans gives stable, interpretable clusters; HDBSCAN is more flexible but produces noisy cluster counts at this scale. |
| Suggested replies  | Groq Llama 3.3 70B (free tier) with deterministic template fallback | LLM-generated replies are the *one* place an LLM clearly beats classical NLP. Fallback ensures the system stays useful when the key is missing or the API is down. |

### Why not "just call an LLM for everything"

A naive design would prompt an LLM for category + sentiment + reply per ticket.
At 5k tickets/day, even on Groq's free tier that's 150k req/month, and on a paid
provider it's $50–300/month for what an embedding + lexicon does for free. We
reserve the LLM for the **generative** step where it actually wins.

## 2. Data Model

Single normalised table (`tickets`) in SQLite (swap to Postgres in production by
flipping `DB_URL`). Vectors live in Chroma keyed by `ticket_id` so the two stores
stay loosely coupled.

```
tickets
├── ticket_id           (string, unique)
├── timestamp           (datetime, indexed — drives all rolling-window queries)
├── customer_id         (string)
├── channel             (chat / email / web)
├── message             (text)
├── agent_reply         (text, nullable)
├── product             (string)
├── order_value         (float)
├── customer_country    (string)
├── resolution_status   (open / in_progress / resolved / escalated)
├── category            (one of 9 — populated by pipeline)
├── category_confidence (float, 0–1)
├── sentiment           (positive / neutral / negative)
├── sentiment_score     (float, -1..1)
├── frustration_level   (int, 0–4)
├── issue_cluster       (int, nullable — KMeans cluster id)
├── suggested_reply     (text, nullable)
└── processed_at        (datetime)
```

Two stores deliberately:
- **SQL** for aggregations (group by category, sum revenue, daily trend).
- **Vector (Chroma)** for similarity search (agent assist, "tickets like this one").

## 3. Pipeline

```
raw row ─► clean ─► embed ─► classify ──┐
                       │                ├─► persist (SQL + Chroma)
                       └─► cluster ─────┘
            sentiment / frustration ─►──┘
```

Idempotent. Re-uploading the same `ticket_id` updates enrichment in place rather
than inserting a duplicate. Clustering only runs when batch size ≥ 24 — small
incremental uploads inherit the cluster assignment from the last full run.

### Batch vs streaming

The same `process_records(rows)` function powers:
- the **batch CLI** (`python -m scripts.run_pipeline --csv ...`),
- the **HTTP bulk** endpoint (`POST /tickets/upload`),
- the **single-ticket** endpoint (`POST /tickets`),
- and a future **queue worker** (e.g. RabbitMQ / SQS consumer that calls
  `process_records([msg])`).

No code path is duplicated between batch and streaming.

## 4. Scalability

| Concern                          | Today (demo)                                   | At 50k tickets/day                                                                  |
| -------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| Storage                          | SQLite file                                    | Managed Postgres (RDS / Cloud SQL). Schema is unchanged.                            |
| Vector store                     | Chroma (filesystem)                            | Qdrant / Weaviate cluster or pgvector. Same upsert API behind a thin adapter.       |
| Embedding throughput             | CPU, ~400 msgs/sec                             | Move to an inference service (TEI, BentoML) on a GPU node or a managed embedding API. |
| LLM rate limit                   | Groq free tier (~30 RPM)                       | Paid Groq / dedicated Bedrock endpoint; fall back to templates on 429.              |
| Clustering                       | KMeans batch nightly                           | MiniBatchKMeans on streaming windows, or BERTopic for topic-stability over time.     |
| API                              | Single uvicorn process on Render free          | Horizontal: stateless containers behind a load balancer; shared DB; sticky-free.    |
| Pipeline orchestration           | Direct fn calls                                | Move heavy stages to a queue (Celery / Arq / Cloud Tasks).                          |

## 5. Trade-offs

- **Synthetic data > real Kaggle dump.** Picking synthesis means we control the
  category distribution and have ground-truth labels for evaluation, at the cost
  of "realism". A future iteration can mix in 1–2 public dumps for stress
  testing.
- **SQLite default.** Zero-infra to run, but file-locking limits concurrent
  writes. Production swap to Postgres is a one-liner (`DB_URL`).
- **Streamlit dashboard.** Built in hours, talks to the API over HTTP. Less
  polished than React but matches the "lightweight dashboard" requirement and
  keeps the demo dependency-free.
- **Free-tier LLM.** Groq Llama 3.3 70B is fast and free but rate-limited and
  occasionally less coherent than a frontier model. The fallback template path
  means the system degrades gracefully rather than failing closed.
- **Frustration as a custom score, not a model.** Cheap and explainable, but
  brittle for sarcasm and indirect anger. A finetuned classifier on real
  ticket data would be the next step once we have labelled examples.
