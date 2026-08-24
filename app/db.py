"""SQLite persistence for candidates, evaluations, and cost telemetry."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy import Column, JSON, Float, Integer, String, Text

logger = logging.getLogger(__name__)

DB_PATH: Path = Path(__file__).resolve().parent.parent / "screener.db"
DATABASE_URL: str = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)


class Candidate(SQLModel, table=True):
    """Persisted resume record after parse/anonymization."""

    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(sa_column=Column(String(512), nullable=False))
    anonymized_text: str = Field(default="", sa_column=Column(Text, nullable=False))
    raw_text: str = Field(default="", sa_column=Column(Text, nullable=False))
    parsed_json: Dict[str, Any] = Field(
        default_factory=dict,
        sa_column=Column(JSON, nullable=False),
    )


class Evaluation(SQLModel, table=True):
    """Hybrid scoring result for a candidate against a job description."""

    id: Optional[int] = Field(default=None, primary_key=True)
    candidate_id: int = Field(sa_column=Column(Integer, nullable=False, index=True))
    job_description: str = Field(default="", sa_column=Column(Text, nullable=False))
    match_score: float = Field(default=0.0, sa_column=Column(Float, nullable=False))
    vector_score: float = Field(default=0.0, sa_column=Column(Float, nullable=False))
    llm_score: float = Field(default=0.0, sa_column=Column(Float, nullable=False))
    skill_gaps: List[str] = Field(default_factory=list, sa_column=Column(JSON, nullable=False))
    token_cost: float = Field(default=0.0, sa_column=Column(Float, nullable=False))
    latency_seconds: float = Field(default=0.0, sa_column=Column(Float, nullable=False))


def init_db() -> None:
    """Create SQLite tables if they do not already exist."""
    try:
        SQLModel.metadata.create_all(engine)
        logger.info("Database initialized at %s", DB_PATH)
    except Exception as exc:
        logger.exception("Failed to initialize database: %s", exc)
        raise


def save_candidate(
    filename: str,
    anonymized_text: str,
    raw_text: str,
    parsed_json: Optional[Dict[str, Any]] = None,
) -> Candidate:
    """Insert a candidate row and return the persisted model with id."""
    init_db()
    payload: Dict[str, Any] = parsed_json if parsed_json is not None else {}
    try:
        candidate = Candidate(
            filename=filename,
            anonymized_text=anonymized_text,
            raw_text=raw_text,
            parsed_json=payload,
        )
        with Session(engine) as session:
            session.add(candidate)
            session.commit()
            session.refresh(candidate)
            return candidate
    except Exception as exc:
        logger.exception("Failed to save candidate '%s': %s", filename, exc)
        raise


def save_evaluation(
    candidate_id: int,
    job_description: str,
    match_score: float,
    vector_score: float,
    llm_score: float,
    skill_gaps: Optional[List[str]] = None,
    token_cost: float = 0.0,
    latency_seconds: float = 0.0,
) -> Evaluation:
    """Insert an evaluation row for a candidate and return the persisted model."""
    init_db()
    gaps: List[str] = skill_gaps if skill_gaps is not None else []
    try:
        evaluation = Evaluation(
            candidate_id=candidate_id,
            job_description=job_description,
            match_score=match_score,
            vector_score=vector_score,
            llm_score=llm_score,
            skill_gaps=gaps,
            token_cost=token_cost,
            latency_seconds=latency_seconds,
        )
        with Session(engine) as session:
            session.add(evaluation)
            session.commit()
            session.refresh(evaluation)
            return evaluation
    except Exception as exc:
        logger.exception("Failed to save evaluation for candidate %s: %s", candidate_id, exc)
        raise


def get_all_candidates() -> List[Candidate]:
    """Return every stored candidate, newest first."""
    init_db()
    try:
        with Session(engine) as session:
            statement = select(Candidate).order_by(Candidate.id.desc())
            return list(session.exec(statement).all())
    except Exception as exc:
        logger.exception("Failed to load candidates: %s", exc)
        return []


def get_latest_evaluation(candidate_id: int) -> Optional[Evaluation]:
    """Return the most recent evaluation for a candidate, if any."""
    init_db()
    try:
        with Session(engine) as session:
            statement = (
                select(Evaluation)
                .where(Evaluation.candidate_id == candidate_id)
                .order_by(Evaluation.id.desc())
            )
            return session.exec(statement).first()
    except Exception as exc:
        logger.exception("Failed to load evaluation for candidate %s: %s", candidate_id, exc)
        return None


def get_all_evaluations() -> List[Evaluation]:
    """Return all evaluations for analytics aggregation."""
    init_db()
    try:
        with Session(engine) as session:
            statement = select(Evaluation).order_by(Evaluation.id.desc())
            return list(session.exec(statement).all())
    except Exception as exc:
        logger.exception("Failed to load evaluations: %s", exc)
        return []


def candidate_to_dict(candidate: Candidate) -> Dict[str, Any]:
    """Serialize a Candidate SQLModel instance to a JSON-safe dict."""
    parsed: Any = candidate.parsed_json
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            parsed = {"raw": parsed}
    return {
        "id": candidate.id,
        "filename": candidate.filename,
        "anonymized_text": candidate.anonymized_text,
        "raw_text": candidate.raw_text,
        "parsed_json": parsed if isinstance(parsed, dict) else {},
    }
