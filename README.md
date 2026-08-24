# Smart Resume Screener & Recruiter Co-Pilot

Production-style screening stack: FastAPI + Pydantic v2 + SentenceTransformers + ChromaDB + SQLite (SQLModel) + Streamlit.

The pipeline extracts PDF text, optionally redacts PII, scores each resume with a hybrid **40% embedding / 60% LLM** formula, stores results in SQLite, indexes chunks in ChromaDB, and exposes a recruiter chat co-pilot plus cost analytics.

## Architecture

| Layer | Module | Role |
| --- | --- | --- |
| API | `app/main.py` | REST: `/upload`, `/query`, `/analytics` |
| Parse | `app/parser.py` | `pdfplumber` extraction + regex PII strip |
| Score | `app/engine.py` | MiniLM cosine similarity + structured LLM JSON |
| RAG | `app/rag.py` | Persistent Chroma collection at `./chroma_db` |
| Store | `app/db.py` | SQLite `screener.db` (`Candidate`, `Evaluation`) |
| UI | `frontend/app.py` | Dark Streamlit dashboard (3 tabs) |

Hybrid score:

```text
final_score = (vector_score * 0.4) + ((llm_score * 10) * 0.6)
```

`vector_score` is 0–100. `llm_score` is 1.0–10.0. Token cost is estimated at **$0.0015 / 1k tokens**. If `OPENAI_API_KEY` is missing or the call fails, a deterministic keyword scorer is used and spend stays `$0`.

## Setup

```bash
cd smart-resume-screener
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` if you have an OpenAI key. The first run downloads `sentence-transformers/all-MiniLM-L6-v2`.

## Run

Terminal 1 — API:

```bash
cd smart-resume-screener
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — UI:

```bash
cd smart-resume-screener
source .venv/bin/activate
streamlit run frontend/app.py
```

Open the Streamlit URL (typically `http://localhost:8501`). The UI talks to FastAPI when `/health` is up; otherwise it runs the same pipeline in-process.

## API

- `POST /upload` — multipart: `files` (PDFs), `job_description`, `anonymize`
- `POST /query` — JSON `{ "query": "...", "top_k": 5 }`
- `GET /analytics` — candidate count, average match, total latency, cumulative token cost
- `GET /health` — liveness

Example:

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "job_description=Senior Python engineer with FastAPI and RAG experience" \
  -F "anonymize=true" \
  -F "files=@resume.pdf"
```

## Dashboard tabs

1. **Candidate Screening** — sidebar JD + PII toggle, multi-PDF upload, ranked table, score bars, skill-gap tags, justification.
2. **Recruiter Co-Pilot (Chat)** — `st.chat_message` over Chroma snippets.
3. **System & Cost Analytics** — resumes processed, average latency, LLM spend.

## Notes

- Image-only PDFs have no extractable text and are rejected with a clear error.
- SQLite file: `screener.db` in the project root. Vector store: `chroma_db/`.
- FastAPI file uploads require `python-multipart` (already listed in `requirements.txt`).
