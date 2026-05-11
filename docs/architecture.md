# Architecture

## High-level flow

```
                    ┌──────────────────────────┐
                    │   Ticket sources         │
                    │  (chat / email / web)    │
                    └──────────┬───────────────┘
                               │
                               ▼
                   ┌────────────────────────┐
                   │   FastAPI backend      │
                   │   - POST /tickets      │
                   │   - POST /tickets/bulk │
                   │   - POST /tickets/upload (csv) │
                   │   - POST /tickets/reply        │
                   │   - GET  /insights/summary     │
                   └──────────┬─────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
   ┌────────────────────┐          ┌────────────────────┐
   │  Pipeline (sync)   │          │  Suggested-reply   │
   │  ─ clean           │          │  Groq Llama 3.3    │
   │  ─ embed (MiniLM)  │          │  + template        │
   │  ─ classify (0-shot│          │  fallback          │
   │  ─ sentiment (VADER)          └──────────┬─────────┘
   │  ─ frustration     │                     │
   │  ─ cluster (KMeans)│                     │
   └─────────┬──────────┘                     │
             │                                 │
   ┌─────────▼──────────┐    ┌────────────────▼────────┐
   │  SQLite / Postgres │    │  ChromaDB (vectors)     │
   │  tickets table     │    │  semantic search        │
   └────────────────────┘    └─────────────────────────┘

                              ▲
                              │
                   ┌──────────┴──────────────┐
                   │  Streamlit dashboard    │
                   │  Overview / Top issues /│
                   │  Tickets / Agent Assist │
                   │  / Upload               │
                   └─────────────────────────┘
```

## Container topology

```
        ┌─────────────┐    HTTP    ┌─────────────┐
        │ dashboard   │  ────────► │ api         │
        │ (Streamlit) │            │ (FastAPI)   │
        │  :8501      │  ◄──────── │  :8000      │
        └─────────────┘   JSON     └──────┬──────┘
                                          │
                                  ┌───────┴───────┐
                                  │  app-data     │  (named volume)
                                  │  ├─ tickets.db│
                                  │  └─ chroma/   │
                                  └───────────────┘
```

Locally: `docker compose up` brings both services up.
In production on Render: both services are provisioned via
[`render.yaml`](../render.yaml). The dashboard talks to the API over the
public HTTPS URL Render assigns to the API service.

## CI/CD

```
  push / PR  ──►  GH Actions
                  ├─ pytest (no LLM, fallback path)
                  ├─ docker build (api + dashboard)
                  └─ on main → curl Render deploy hooks (API + dashboard)
```

The `deploy` job only runs on `main` and only after both `test` and `docker`
pass. Secrets needed: `RENDER_DEPLOY_HOOK_API`,
`RENDER_DEPLOY_HOOK_DASHBOARD`. `GROQ_API_KEY` is set in the Render UI
directly — it never enters CI.
