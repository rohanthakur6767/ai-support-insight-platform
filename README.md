# AI-Powered Customer Support Insight Platform

Ingest customer-support tickets, surface what's broken, suggest replies. Built
for the AI/ML-Software Dev assignment.

- **Backend:** FastAPI + SQLite + ChromaDB
- **AI:** sentence-transformers (`all-MiniLM-L6-v2`), VADER, KMeans, Groq Llama 3.3 70B
- **Dashboard:** Streamlit + Plotly
- **Deploy:** Docker + docker-compose, Render (free tier), GitHub Actions CI/CD

> Design rationale: [`docs/DESIGN.md`](docs/DESIGN.md) &middot;
> Business writeup: [`docs/BUSINESS.md`](docs/BUSINESS.md) &middot;
> Architecture diagram: [`docs/architecture.md`](docs/architecture.md)

---

## Quick start (local, no Docker)

```bash
# 1. dependencies
python -m pip install -r requirements.txt

# 2. config — get a free key at https://console.groq.com
cp .env.example .env
# edit .env and set GROQ_API_KEY (optional; works without it via fallbacks)

# 3. generate the synthetic dataset (5k tickets across 9 categories)
python -m scripts.generate_data --n 5000 --out data/tickets.csv

# 4. run the AI pipeline (clean → embed → classify → sentiment → cluster → store)
python -m scripts.run_pipeline --csv data/tickets.csv

# 5. start the API
uvicorn app.main:app --reload --port 8000
# → http://localhost:8000/docs

# 6. in another shell, start the dashboard
streamlit run dashboard/app.py
# → http://localhost:8501
```

## Quick start (Docker)

```bash
cp .env.example .env  # fill in GROQ_API_KEY if you have one
docker compose up --build
# api  → http://localhost:8000/docs
# dash → http://localhost:8501
```

Seed the running API with the synthetic dataset:

```bash
python -m scripts.generate_data --n 5000
curl -F "file=@data/tickets.csv" http://localhost:8000/tickets/upload
```

## Endpoints

| Method | Path                         | Purpose                                   |
| ------ | ---------------------------- | ----------------------------------------- |
| POST   | `/tickets`                   | Submit a single ticket                    |
| POST   | `/tickets/bulk`              | Submit a JSON array of tickets            |
| POST   | `/tickets/upload`            | Upload a CSV file                         |
| GET    | `/tickets`                   | List with filters                         |
| GET    | `/tickets/{ticket_id}`       | Single ticket                             |
| GET    | `/tickets/search/semantic`   | Vector search across messages             |
| POST   | `/tickets/reply`             | Suggested-reply (LLM) for a message       |
| GET    | `/insights/summary`          | KPIs + category mix + trend + top issues  |
| GET    | `/insights/revenue-by-category` | Revenue exposure per category           |
| GET    | `/insights/volume-by-day`    | Ticket volume time series                 |
| GET    | `/health`                    | Liveness probe                            |
| GET    | `/docs`                      | OpenAPI / Swagger UI                      |

## Project layout

```
app/
├── main.py             FastAPI app + router wiring
├── config.py           pydantic-settings (env-driven)
├── api/
│   ├── tickets.py      Ingestion + retrieval + reply
│   └── insights.py     Aggregations
├── pipeline/
│   ├── clean.py        Text normalisation
│   ├── embed.py        sentence-transformers + Chroma
│   ├── classify.py     Zero-shot category classifier
│   ├── sentiment.py    VADER + frustration heuristic
│   ├── issues.py       KMeans top-issue clustering
│   ├── reply.py        Groq LLM (+ template fallback)
│   └── runner.py       End-to-end orchestrator
├── models/
│   ├── db.py           SQLAlchemy schema + session
│   └── schemas.py      Pydantic I/O
└── data/synthesize.py  Synthetic dataset generator
dashboard/app.py        Streamlit UI
scripts/                CLI entrypoints
tests/                  pytest smoke tests
docs/                   Design, business, architecture
.github/workflows/ci.yml CI/CD (test → docker → render deploy hook)
Dockerfile / Dockerfile.dashboard / docker-compose.yml
render.yaml             Render Blueprint (API + dashboard, free tier)
```

## Testing

```bash
pytest -q
```

Tests run without an LLM (the reply module falls back to deterministic
templates when `GROQ_API_KEY` is empty), so CI is self-contained.

## Deploy to Render (free tier)

1. Push this repo to GitHub.
2. In the Render dashboard: **New +** → **Blueprint** → pick this repo. The
   [`render.yaml`](render.yaml) provisions two services (API + dashboard).
3. On the **support-insights-api** service → *Environment* tab, set
   `GROQ_API_KEY` (kept out of git).
4. On the **support-insights-dashboard** service → *Environment* tab, set
   `API_BASE_URL` to the API's public URL
   (e.g. `https://support-insights-api.onrender.com`).
5. Optional CI/CD: copy each service's *Deploy Hook* URL into GitHub Actions
   secrets (`RENDER_DEPLOY_HOOK_API`, `RENDER_DEPLOY_HOOK_DASHBOARD`). The
   `deploy` job will then redeploy both services on every push to `main`.

> **Free tier caveats**
> - No persistent disk → the API regenerates 5k synthetic tickets at first
>   boot (~30–60 s). Toggle off via `SEED_ON_BOOT=0`.
> - Services spin down after 15 min idle → first request after idle is a
>   ~30 s cold start.

## License

MIT.
