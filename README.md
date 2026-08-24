
# Smart Resume Screener & Recruiter Co-Pilot

🎥 **[Watch the Live Application Demo Video](https://drive.google.com/file/d/1XCBl0JDA_qA_o-6UHJ8Q9FZFxxYUYrnv/view?usp=sharing)**

A production-style candidate screening stack powered by **FastAPI**, **Pydantic v2**, **SentenceTransformers**, **ChromaDB**, **SQLite (SQLModel)**, and **Streamlit**.

The pipeline extracts PDF text, optionally redacts PII for bias reduction, scores each resume using a hybrid ranking formula (40% vector similarity / 60% LLM evaluation), stores candidate records in SQLite, indexes text chunks in ChromaDB, and exposes an interactive recruiter chat co-pilot alongside enterprise cost analytics.

---

## Architecture Overview

| Layer | Module | Role |
| --- | --- | --- |
| **API** | `app/main.py` | REST endpoints: `/upload`, `/query`, `/analytics`, `/health` |
| **Parse** | `app/parser.py` | `pdfplumber` text extraction + regex-based PII redaction |
| **Score** | `app/engine.py` | MiniLM cosine similarity + structured LLM JSON scoring |
| **RAG** | `app/rag.py` | Persistent Chroma collection stored at `./chroma_db` |
| **Store** | `app/db.py` | SQLite database (`screener.db`) tracking Candidates and Evaluations |
| **UI** | `frontend/app.py` | Streamlit dashboard with warm light theme and 3 functional tabs |

### Scoring Methodology

The final candidate rank is calculated using a hybrid formula:

$$\text{Final Score} = (\text{Vector Score} \times 0.4) + ((\text{LLM Score} \times 10) \times 0.6)$$

* **Vector Score:** Normalized range $0 - 100$ based on cosine similarity.
* **LLM Score:** Rated $1.0 - 10.0$ via structured JSON responses.
* **Cost Efficiency:** Token expenditure is tracked at ~$0.0015 / 1k tokens. If `OPENAI_API_KEY` is missing or fails, the system automatically falls back to a deterministic keyword scorer ($0 cost).

---

## Quick Start

### 1. Setup Environment

```bash
# Navigate to project directory
cd smart-resume-screener

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env

```

*(Note: On first run, `sentence-transformers/all-MiniLM-L6-v2` will download automatically.)*

---

### 2. Running the Application

**Terminal 1 — FastAPI Backend:**

```bash
cd smart-resume-screener
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

```

**Terminal 2 — Streamlit Frontend:**

```bash
cd smart-resume-screener
source .venv/bin/activate
streamlit run frontend/app.py

```

Open `http://localhost:8501` in your browser. The UI seamlessly connects to FastAPI when active, falling back to local in-process execution if offline.

---

## API Documentation

* `POST /upload` — Multipart form upload for candidate PDFs, job description, and PII anonymization toggle.
* `POST /query` — JSON payload (`{"query": "...", "top_k": 5}`) to retrieve relevant candidate chunks via ChromaDB RAG.
* `GET /analytics` — Returns metrics including total candidate count, average match score, latency, and cumulative LLM token spend.
* `GET /health` — Service liveness probe.

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "job_description=Senior Python engineer with FastAPI and RAG experience" \
  -F "anonymize=true" \
  -F "files=@resume.pdf"

```

---

## Dashboard Features

* **Candidate Screening:** Full-width Job Description input, PDF uploader, match score progress bars, extracted skill-gap tags, and detailed justifications.
* **Recruiter Co-Pilot (Chat):** Conversational RAG interface powered by ChromaDB snippet retrieval.
* **System & Cost Analytics:** High-level telemetry for processing latency, total token consumption, and live API status.

---

### Push Updated README to GitHub:

Run this in your terminal to update GitHub:

```bash
git add README.md
git commit -m "docs: add comprehensive README documentation"
git push origin main

```
