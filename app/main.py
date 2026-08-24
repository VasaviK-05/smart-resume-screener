"""FastAPI application: upload, score, query, and analytics endpoints."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.db import (
    candidate_to_dict,
    get_all_candidates,
    get_all_evaluations,
    get_latest_evaluation,
    init_db,
    save_candidate,
    save_evaluation,
)
from app.engine import evaluate_matches, warm_models
from app.parser import parse_resume
from app.rag import index_resumes, query_resumes

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    warm_models()
    logger.info("Smart Resume Screener API ready")
    yield


app = FastAPI(
    title="Smart Resume Screener & Recruiter Co-Pilot",
    description="Hybrid semantic + LLM screening with PII anonymization and RAG chat.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language recruiter question")
    top_k: int = Field(default=5, ge=1, le=20)


class QueryHit(BaseModel):
    candidate_id: int
    filename: str
    chunk_index: int
    snippet: str
    similarity: float


class QueryResponse(BaseModel):
    query: str
    matches: List[QueryHit]


class AnalyticsResponse(BaseModel):
    total_candidates: int
    average_match_score: float
    total_latency_seconds: float
    cumulative_token_cost: float
    openai_configured: bool


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_resumes(
    job_description: str = Form(..., min_length=10),
    anonymize: str = Form("true"),
    files: List[UploadFile] = File(...),
) -> Dict[str, Any]:
    """Parse, anonymize, score, persist, and index one or more PDF resumes."""
    anonymize_flag: bool = _as_bool(anonymize)
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF file is required.")

    parsed: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for upload in files:
        filename: str = upload.filename or "unnamed.pdf"
        if not filename.lower().endswith(".pdf"):
            errors.append({"filename": filename, "error": "Only PDF files are supported."})
            continue
        try:
            pdf_bytes: bytes = await upload.read()
            raw_text, anonymized_text, metadata = parse_resume(pdf_bytes, anonymize=anonymize_flag)
            scoring_text: str = anonymized_text if anonymize_flag else raw_text
            parsed.append(
                {
                    "filename": filename,
                    "raw_text": raw_text,
                    "anonymized_text": anonymized_text,
                    "scoring_text": scoring_text,
                    "metadata": metadata,
                }
            )
        except ValueError as exc:
            logger.warning("Rejected file %s: %s", filename, exc)
            errors.append({"filename": filename, "error": str(exc)})
        except Exception as exc:
            logger.exception("Failed parsing %s: %s", filename, exc)
            errors.append({"filename": filename, "error": f"Processing failed: {exc}"})

    results: List[Dict[str, Any]] = []
    if parsed:
        evaluations: List[Dict[str, Any]] = evaluate_matches(
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
                persisted = save_evaluation(
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
                        "evaluation_id": persisted.id,
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
                        "anonymized": anonymize_flag,
                        "preview": item["scoring_text"][:800],
                    }
                )
            except Exception as exc:
                logger.exception("Failed persisting %s: %s", item["filename"], exc)
                errors.append({"filename": item["filename"], "error": f"Processing failed: {exc}"})

    results.sort(key=lambda row: float(row.get("match_score", 0.0)), reverse=True)
    if not results and errors:
        raise HTTPException(status_code=422, detail={"message": "No resumes could be processed.", "errors": errors})

    return {"count": len(results), "results": results, "errors": errors}


@app.post("/query", response_model=QueryResponse)
def recruiter_query(payload: QueryRequest) -> QueryResponse:
    """Retrieve top matching resume snippets for a recruiter question."""
    try:
        hits: List[Dict[str, Any]] = query_resumes(payload.query, top_k=payload.top_k)
        return QueryResponse(query=payload.query, matches=[QueryHit(**hit) for hit in hits])
    except Exception as exc:
        logger.exception("Query endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@app.get("/analytics", response_model=AnalyticsResponse)
def analytics() -> AnalyticsResponse:
    """Aggregate candidate volume, average score, latency, and token spend."""
    try:
        candidates = get_all_candidates()
        evaluations = get_all_evaluations()
        scores: List[float] = [float(item.match_score) for item in evaluations]
        latencies: List[float] = [float(item.latency_seconds) for item in evaluations]
        costs: List[float] = [float(item.token_cost) for item in evaluations]
        average: float = round(sum(scores) / len(scores), 2) if scores else 0.0
        return AnalyticsResponse(
            total_candidates=len(candidates),
            average_match_score=average,
            total_latency_seconds=round(sum(latencies), 4),
            cumulative_token_cost=round(sum(costs), 6),
            openai_configured=bool(os.getenv("OPENAI_API_KEY")),
        )
    except Exception as exc:
        logger.exception("Analytics endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Analytics failed: {exc}") from exc


@app.get("/candidates")
def list_candidates() -> Dict[str, Any]:
    """Debug/helper listing of stored candidates with latest evaluation."""
    rows: List[Dict[str, Any]] = []
    for candidate in get_all_candidates():
        record: Dict[str, Any] = candidate_to_dict(candidate)
        evaluation = get_latest_evaluation(candidate.id or 0)
        record["latest_evaluation"] = evaluation.model_dump() if evaluation is not None else None
        rows.append(record)
    return {"candidates": rows}
