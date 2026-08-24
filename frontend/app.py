"""Streamlit dashboard for Smart Resume Screener & Recruiter Co-Pilot."""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

ROOT: Path = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

API_URL: str = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS: int = 180

st.set_page_config(
    page_title="Recruiter Dashboard",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
/* Base Canvas - Soft Warm Beige */
.stApp, [data-testid="stAppViewContainer"], .main {
    background-color: #F7F5F0 !important;
    color: #1A1A1A !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Remove default borders & header noise */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Hide Streamlit sidebar chrome */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

.backend-banner {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    background: #FFFFFF;
    border: 1px solid #E2DEC9;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 0 0 1.1rem 0;
    color: #1A1A1A;
    font-size: 0.9rem;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.02);
}
.backend-banner .backend-value {
    color: #1A1A1A;
    font-weight: 600;
}
.backend-banner .backend-url {
    color: #666660;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.85rem;
}

/* Tab Bar Styling */
button[data-baseweb="tab"] {
    color: #666660 !important;
    font-weight: 500 !important;
    font-size: 15px !important;
}
button[aria-selected="true"] {
    color: #1A1A1A !important;
    font-weight: 700 !important;
    border-bottom-color: #1A1A1A !important;
}

/* Job Description Textarea */
textarea {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1px solid #E2DEC9 !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.02) !important;
}
textarea::placeholder {
    color: #8C887B !important;
}

/* File Uploader Container - Clean Off-White Box */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 2px dashed #D6D0C1 !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
}
[data-testid="stFileUploader"] section {
    background-color: #FAF8F5 !important;
}
[data-testid="stFileUploader"] * {
    color: #1A1A1A !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: #666660 !important;
}
[data-testid="stFileUploader"] button {
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: none !important;
}
[data-testid="stFileUploader"] button * {
    color: #FFFFFF !important;
}

/* Process Candidates Button - Crisp Warm Black */
button[kind="primary"], button[data-testid="baseButton-primary"] {
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    font-size: 15px !important;
    transition: all 0.2s ease !important;
}
button[kind="primary"]:hover {
    background-color: #333333 !important;
}
button[kind="primary"] *, button[data-testid="baseButton-primary"] * {
    color: #FFFFFF !important;
}

/* Alert / Info Banner */
[data-testid="stNotification"] {
    background-color: #EFECE6 !important;
    border: 1px solid #DDD8CE !important;
    color: #4A473E !important;
    border-radius: 10px !important;
}

/* Streamlit Chat Input Box Fix */
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #E2DEC9 !important;
    border-radius: 12px !important;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.02) !important;
}

[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #1A1A1A !important;
    border: none !important;
}

[data-testid="stChatInput"] button {
    background-color: #1A1A1A !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] button * {
    color: #FFFFFF !important;
}

/* Labels & Text Spans */
label, p, span, div {
    color: #1A1A1A;
}
</style>
""",
    unsafe_allow_html=True,
)

DASHBOARD_CSS: str = """
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding-top: 1.6rem; max-width: 1120px; }

    h1, h2, h3 { color: #1A1A1A !important; font-weight: 600 !important; letter-spacing: 0.04em; }
    h1 { font-size: 1.9rem !important; letter-spacing: -0.03em !important; }
    h2, h3 { font-size: 0.78rem !important; letter-spacing: 0.18em !important; text-transform: uppercase !important; color: #666660 !important; }
    .hero-kicker {
        display: inline-block;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #666660;
        margin-bottom: 0.4rem;
    }
    .hero-sub { color: #666660; font-size: 0.95rem; margin: 0 0 1.4rem 0; }

    .glass-card, .candidate-card, .board-wrap, .card-body, .metric-tile, .justification-box, .analytics-tile {
        background: #FFFFFF;
        border: 1px solid #E2DEC9;
        border-radius: 14px;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.02);
    }
    .candidate-card { padding: 1.3rem 1.35rem 1.25rem 1.35rem; margin: 0 0 1.2rem 0; }
    .card-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .card-title { font-weight: 600; font-size: 1.05rem; color: #1A1A1A; letter-spacing: -0.02em; }
    .score-badge {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #1A1A1A;
        font-variant-numeric: tabular-nums;
    }

    .metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 0 0 1.1rem 0; }
    .metric-tile { padding: 14px 16px 13px 16px; }
    .metric-label {
        display: block;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #666660;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-weight: 700;
        font-size: 24px;
        letter-spacing: -0.04em;
        color: #1A1A1A;
        font-variant-numeric: tabular-nums;
        line-height: 1.1;
    }

    .match-bar-wrap { margin: 0.15rem 0 1.1rem 0; }
    .match-bar-pct {
        font-size: 24px;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: #1A1A1A;
        margin-bottom: 10px;
        font-variant-numeric: tabular-nums;
        line-height: 1.1;
    }
    .match-track {
        background: rgba(0, 0, 0, 0.08);
        height: 10px;
        border-radius: 8px;
        overflow: hidden;
    }
    .match-fill {
        height: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #1A1A1A 0%, #8A8474 100%);
        transform-origin: left center;
        animation: growBar 0.75s cubic-bezier(0.22, 1, 0.36, 1);
    }
    @keyframes growBar {
        from { transform: scaleX(0); }
        to { transform: scaleX(1); }
    }

    .section-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #666660;
        margin: 0.15rem 0 0.5rem 0;
    }
    .badge-strength, .badge-gap, .badge-strength, .badge-gap {
        display: inline-block;
        border-radius: 20px;
        padding: 4px 10px;
        margin: 0 6px 6px 0;
        font-size: 12px;
        font-weight: 500;
    }
    .badge-strength, .badge-strength {
        background: rgba(52, 199, 89, 0.12);
        border: 1px solid #34C759;
        color: #248A3D;
    }
    .badge-gap, .badge-gap {
        background: rgba(255, 59, 48, 0.12);
        border: 1px solid #FF3B30;
        color: #D70015;
    }

    .justification-box {
        margin: 0.85rem 0 0 0;
        padding: 0.85rem 1rem 0.9rem 1rem;
        border-left: 2px solid #1A1A1A;
        color: #1A1A1A;
        line-height: 1.6;
        font-size: 0.9rem;
    }
    .justification-box strong {
        display: block;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #666660;
        margin-bottom: 0.45rem;
    }

    .board-wrap { overflow: hidden; margin: 0.35rem 0 1.4rem 0; }
    table.leaderboard { width: 100%; border-collapse: collapse; font-size: 0.88rem; background: #FFFFFF; }
    table.leaderboard thead th {
        background: #FAF8F5;
        color: #666660;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-size: 11px;
        padding: 14px 16px;
        border-bottom: 1px solid #E2DEC9;
        text-align: left;
        white-space: nowrap;
    }
    table.leaderboard tbody td {
        padding: 13px 16px;
        color: #1A1A1A;
        border-bottom: 1px solid #E2DEC9;
        font-variant-numeric: tabular-nums;
    }
    table.leaderboard tbody tr:last-child td { border-bottom: 0; }
    table.leaderboard tbody tr:hover { background: #FAF8F5; }
    .cell-score, .cell-vector, .cell-llm { font-weight: 700; color: #1A1A1A; }
    .rank-pill { color: #666660; font-weight: 600; font-size: 0.82rem; }

    .stButton > button {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2DEC9;
        border-radius: 14px;
        padding: 0.95rem 1.05rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        color: #666660 !important;
    }
    [data-testid="stMetricValue"] {
        color: #1A1A1A !important;
        font-weight: 700 !important;
        font-size: 24px !important;
    }
    .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #E2DEC9; }
    .stTabs [data-baseweb="tab"] { background: transparent; border: 0; color: #666660; font-weight: 500; }
    .stTabs [aria-selected="true"] {
        background: transparent !important;
        color: #1A1A1A !important;
        border-bottom: 1px solid #1A1A1A !important;
    }
    [data-testid="stProgress"], [data-testid="stProgressBar"] { display: none !important; }
    .stCaption { color: #666660; }
    label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
        color: #1A1A1A !important;
        font-weight: 600 !important;
    }
</style>
"""

st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)


def _escape(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _badge_row(items: List[Any], css_class: str, empty_label: str) -> str:
    values: List[str] = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        return f"<span class='{css_class}'>{_escape(empty_label)}</span>"
    return "".join(f"<span class='{css_class}'>{_escape(item)}</span>" for item in values)


def _match_bar(score: float) -> str:
    width: float = max(0.0, min(100.0, float(score)))
    return (
        "<div class='match-bar-wrap'>"
        f"<div class='match-bar-pct'>{width:.1f}%</div>"
        "<div class='match-track'>"
        f"<div class='match-fill' style='width:{width:.1f}%'></div>"
        "</div></div>"
    )


def api_available() -> bool:
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def process_locally(
    files: List[Any],
    job_description: str,
    anonymize: bool,
) -> Dict[str, Any]:
    """Fallback path when the FastAPI process is not running."""
    from app.db import save_candidate, save_evaluation
    from app.engine import evaluate_matches
    from app.parser import parse_resume
    from app.rag import index_resumes

    parsed: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for uploaded in files:
        filename: str = uploaded.name
        try:
            raw_text, anonymized_text, metadata = parse_resume(uploaded.getvalue(), anonymize=anonymize)
            scoring_text: str = anonymized_text if anonymize else raw_text
            parsed.append(
                {
                    "filename": filename,
                    "raw_text": raw_text,
                    "anonymized_text": anonymized_text,
                    "scoring_text": scoring_text,
                    "metadata": metadata,
                }
            )
        except Exception as exc:
            errors.append({"filename": filename, "error": str(exc)})

    results: List[Dict[str, Any]] = []
    if parsed:
        evaluations = evaluate_matches(
            [item["scoring_text"] for item in parsed],
            job_description,
        )
        for item, evaluation in zip(parsed, evaluations):
            try:
                candidate = save_candidate(
                    filename=item["filename"],
                    anonymized_text=item["anonymized_text"],
                    raw_text=item["raw_text"],
                    parsed_json={
                        **item["metadata"],
                        "strengths": evaluation["strengths"],
                        "reasoning": evaluation["reasoning"],
                        "source": evaluation["source"],
                    },
                )
                if candidate.id is None:
                    raise RuntimeError("Candidate was not assigned a primary key.")
                save_evaluation(
                    candidate_id=candidate.id,
                    job_description=job_description,
                    match_score=evaluation["final_score"],
                    vector_score=evaluation["vector_score"],
                    llm_score=evaluation["llm_score"],
                    skill_gaps=evaluation["missing_skills"],
                    token_cost=evaluation["token_cost"],
                    latency_seconds=evaluation["latency_seconds"],
                )
                index_resumes([candidate])
                results.append(
                    {
                        "candidate_id": candidate.id,
                        "filename": item["filename"],
                        "match_score": evaluation["final_score"],
                        "match_percentage": evaluation["match_percentage"],
                        "vector_score": evaluation["vector_score"],
                        "vector_percentage": evaluation["vector_percentage"],
                        "llm_score": evaluation["llm_score"],
                        "skill_gaps": evaluation["missing_skills"],
                        "strengths": evaluation["strengths"],
                        "reasoning": evaluation["reasoning"],
                        "token_cost": evaluation["token_cost"],
                        "latency_seconds": evaluation["latency_seconds"],
                        "source": evaluation["source"],
                        "anonymized": anonymize,
                        "preview": item["scoring_text"][:800],
                    }
                )
            except Exception as exc:
                errors.append({"filename": item["filename"], "error": str(exc)})
    results.sort(key=lambda row: float(row.get("match_score", 0.0)), reverse=True)
    return {"count": len(results), "results": results, "errors": errors}


def process_via_api(
    files: List[Any],
    job_description: str,
    anonymize: bool,
) -> Dict[str, Any]:
    multipart: List[tuple[str, tuple[str, bytes, str]]] = []
    for uploaded in files:
        multipart.append(("files", (uploaded.name, uploaded.getvalue(), "application/pdf")))
    data: Dict[str, Any] = {
        "job_description": job_description,
        "anonymize": str(anonymize).lower(),
    }
    response = requests.post(
        f"{API_URL}/upload",
        files=multipart,
        data=data,
        timeout=TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        detail: Any = response.json().get("detail", response.text) if response.content else response.text
        raise RuntimeError(f"API error {response.status_code}: {detail}")
    payload: Dict[str, Any] = response.json()
    return payload


def fetch_analytics() -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_URL}/analytics", timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    try:
        from app.db import get_all_candidates, get_all_evaluations

        candidates = get_all_candidates()
        evaluations = get_all_evaluations()
        scores = [float(item.match_score) for item in evaluations]
        latencies = [float(item.latency_seconds) for item in evaluations]
        costs = [float(item.token_cost) for item in evaluations]
        return {
            "total_candidates": len(candidates),
            "average_match_score": round(sum(scores) / len(scores), 2) if scores else 0.0,
            "total_latency_seconds": round(sum(latencies), 4),
            "cumulative_token_cost": round(sum(costs), 6),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        }
    except Exception:
        return None


def query_copilot(question: str) -> List[Dict[str, Any]]:
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"query": question, "top_k": 5},
            timeout=60,
        )
        if response.status_code == 200:
            return list(response.json().get("matches", []))
    except requests.RequestException:
        pass
    from app.rag import query_resumes

    return query_resumes(question, top_k=5)


if "screening_results" not in st.session_state:
    st.session_state.screening_results = []
if "screening_errors" not in st.session_state:
    st.session_state.screening_errors = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("<div class='hero-kicker'>Enterprise Recruiter Workspace</div>", unsafe_allow_html=True)
st.title("Smart Resume Screener & Recruiter Co-Pilot")
st.markdown(
    "<p class='hero-sub'>Hybrid BM25 + vector + cross-encoder ranking · PII anonymization · ChromaDB co-pilot</p>",
    unsafe_allow_html=True,
)

live: bool = api_available()
status_label: str = "API connected" if live else "Local fallback mode"
st.markdown(
    "<div class='backend-banner'>"
    f"<span class='backend-value'>Backend: {html.escape(status_label)} / </span>"
    f"<span class='backend-url'>{html.escape(API_URL)}</span>"
    "</div>",
    unsafe_allow_html=True,
)

tab_screen, tab_chat, tab_analytics = st.tabs(
    ["Candidate Screening", "Recruiter Co-Pilot (Chat)", "System & Cost Analytics"]
)

with tab_screen:
    job_description: str = st.text_area(
        "Job Description",
        height=180,
        placeholder="Paste the role requirements, must-have skills, and seniority here…",
    )
    upload_col, privacy_col = st.columns([3, 1], gap="large")
    with upload_col:
        uploads = st.file_uploader(
            "Candidate PDF upload",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more resume PDFs.",
        )
    with privacy_col:
        st.markdown("<div style='height: 1.75rem'></div>", unsafe_allow_html=True)
        anonymize: bool = st.toggle("Anonymize PII", value=True)

    process_clicked: bool = st.button("Process Candidates", type="primary")

    if process_clicked:
        if not job_description.strip():
            st.error("Add a job description before processing.")
        elif not uploads:
            st.error("Upload at least one PDF resume.")
        else:
            with st.spinner("Parsing, embedding, and scoring candidates…"):
                try:
                    if live:
                        payload = process_via_api(uploads, job_description, anonymize)
                    else:
                        payload = process_locally(uploads, job_description, anonymize)
                    st.session_state.screening_results = payload.get("results", [])
                    st.session_state.screening_errors = payload.get("errors", [])
                except Exception as exc:
                    st.error(str(exc))

    errors: List[Dict[str, str]] = st.session_state.screening_errors
    results: List[Dict[str, Any]] = st.session_state.screening_results

    if errors:
        for item in errors:
            st.warning(f"{item.get('filename', 'file')}: {item.get('error', 'unknown error')}")

    if results:
        st.subheader("Ranked candidate board")
        board_rows: List[str] = []
        for index, row in enumerate(results):
            match_val: float = float(row.get("match_score") or 0.0)
            vector_val: float = float(row.get("vector_score") or 0.0)
            llm_val: float = float(row.get("llm_score") or 0.0)
            board_rows.append(
                "<tr>"
                f"<td><span class='rank-pill'>{index + 1}</span></td>"
                f"<td>{_escape(row.get('candidate_id'))}</td>"
                f"<td>{_escape(row.get('filename'))}</td>"
                f"<td class='cell-score'>{match_val:.1f}%</td>"
                f"<td class='cell-vector'>{vector_val:.1f}%</td>"
                f"<td class='cell-llm'>{llm_val:.1f}</td>"
                f"<td>{_escape(row.get('latency_seconds'))}</td>"
                f"<td>{_escape(row.get('token_cost'))}</td>"
                f"<td>{_escape(row.get('source'))}</td>"
                "</tr>"
            )
        st.markdown(
            "<div class='board-wrap'><table class='leaderboard'>"
            "<thead><tr>"
            "<th>Rank</th><th>Candidate ID</th><th>Filename</th>"
            "<th>Match %</th><th>Vector %</th><th>LLM Score</th>"
            "<th>Latency (s)</th><th>Token $</th><th>Source</th>"
            "</tr></thead><tbody>"
            + "".join(board_rows)
            + "</tbody></table></div>",
            unsafe_allow_html=True,
        )

        st.subheader("Candidate detail cards")
        for row in results:
            match_score: float = float(row.get("match_score") or 0.0)
            vector_score: float = float(row.get("vector_score") or 0.0)
            llm_score: float = float(row.get("llm_score") or 0.0)
            strength_html: str = _badge_row(list(row.get("strengths") or []), "badge-strength", "None listed")
            gap_html: str = _badge_row(list(row.get("skill_gaps") or []), "badge-gap", "None listed")
            justification: str = _escape(row.get("reasoning") or "No justification available.")
            st.markdown(
                "<div class='candidate-card'>"
                "<div class='card-header'>"
                f"<div class='card-title'>{_escape(row.get('filename'))}</div>"
                f"<span class='score-badge'>{match_score:.1f}%</span>"
                "</div>"
                "<div class='metric-row'>"
                "<div class='metric-tile'><span class='metric-label'>Match Score</span>"
                f"<span class='metric-value'>{match_score:.1f}%</span></div>"
                "<div class='metric-tile'><span class='metric-label'>Vector Density</span>"
                f"<span class='metric-value'>{vector_score:.1f}%</span></div>"
                "<div class='metric-tile'><span class='metric-label'>Reasoning Index</span>"
                f"<span class='metric-value'>{llm_score:.1f}</span></div>"
                "</div>"
                f"{_match_bar(match_score)}"
                f"<div class='section-label'>Strengths</div>{strength_html}"
                f"<div class='section-label'>Skill gaps</div>{gap_html}"
                f"<div class='justification-box'><strong>Justification</strong>{justification}</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Resume preview"):
                st.text(row.get("preview") or "")
    else:
        st.info("Upload PDFs, paste a job description, and click Process Candidates.")

with tab_chat:
    st.subheader("Ask the recruiter co-pilot")
    st.caption("Queries retrieve the most relevant resume snippets from ChromaDB.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question: Optional[str] = st.chat_input("e.g. Who has production Kubernetes experience?")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Searching indexed resumes…"):
                matches = query_copilot(user_question)
            if not matches:
                answer = "No indexed resume snippets matched that question yet. Process candidates first."
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            else:
                lines: List[str] = [
                    f"Found **{len(matches)}** relevant snippet(s):",
                    "",
                ]
                for hit in matches:
                    lines.append(
                        f"- **{hit.get('filename')}** (candidate #{hit.get('candidate_id')}, "
                        f"similarity {hit.get('similarity')}%)\n"
                        f"  {hit.get('snippet')}"
                    )
                answer = "\n".join(lines)
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

with tab_analytics:
    st.subheader("System telemetry")
    metrics = fetch_analytics()
    if metrics is None:
        st.error("Analytics are unavailable.")
    else:
        total: int = int(metrics.get("total_candidates") or 0)
        avg_score: float = float(metrics.get("average_match_score") or 0.0)
        total_latency: float = float(metrics.get("total_latency_seconds") or 0.0)
        spend: float = float(metrics.get("cumulative_token_cost") or 0.0)
        avg_latency: float = (total_latency / total) if total else 0.0
        token_count: int = int(round((spend / 0.0015) * 1000)) if spend > 0 else 0
        openai_status: str = "Yes" if metrics.get("openai_configured") else "No"

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Resumes Processed", f"{total}")
        col_b.metric("Average Process Time", f"{avg_latency:.1f}s")
        col_c.metric("Total LLM Tokens", f"{token_count:d}")

        extra_a, extra_b, extra_c = st.columns(3)
        extra_a.metric("Average Match Score", f"{avg_score:.1f}%")
        extra_b.metric("LLM Token Spend", f"${spend:.4f}")
        extra_c.metric("OpenAI Configured", openai_status)

        st.caption(
            "Latency is cumulative engine time across evaluations. "
            "Token spend uses $0.0015 / 1k tokens. Mock evaluations record $0.00."
        )
