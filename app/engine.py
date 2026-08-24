"""Hybrid resume-to-JD scorer: SentenceTransformers cosine similarity + LLM structured eval."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

COST_PER_1K_TOKENS: float = 0.0015
VECTOR_WEIGHT: float = 0.4
LLM_WEIGHT: float = 0.6
RRF_K: int = 60
BI_ENCODER_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_embedder: Any = None
_cross_encoder: Any = None
_cross_encoder_failed: bool = False

# Canonical skill -> aliases. Matched with word boundaries so "ml" does not hit "html".
_SKILL_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Python", ("python", "python3")),
    ("Java", ("java",)),
    ("JavaScript", ("javascript", "ecmascript")),
    ("TypeScript", ("typescript", "ts")),
    ("SQL", ("sql", "t-sql", "tsql", "pl/sql", "plsql")),
    ("PostgreSQL", ("postgresql", "postgres")),
    ("MySQL", ("mysql", "mariadb")),
    ("MongoDB", ("mongodb", "mongo")),
    ("Redis", ("redis",)),
    ("Elasticsearch", ("elasticsearch", "opensearch")),
    ("FastAPI", ("fastapi",)),
    ("Flask", ("flask",)),
    ("Django", ("django",)),
    ("Streamlit", ("streamlit",)),
    ("React", ("react", "reactjs", "react.js")),
    ("Node.js", ("node.js", "nodejs", "node")),
    ("Next.js", ("next.js", "nextjs")),
    ("Vue", ("vue", "vuejs", "vue.js")),
    ("Angular", ("angular",)),
    ("HTML", ("html", "html5")),
    ("CSS", ("css", "css3")),
    ("AWS", ("aws", "amazon web services")),
    ("GCP", ("gcp", "google cloud", "google cloud platform")),
    ("Azure", ("azure", "microsoft azure")),
    ("Docker", ("docker",)),
    ("Kubernetes", ("kubernetes", "k8s")),
    ("Terraform", ("terraform",)),
    ("CI/CD", ("ci/cd", "cicd", "github actions", "gitlab ci")),
    ("Linux", ("linux", "unix")),
    ("Git", ("git", "github", "gitlab")),
    ("REST", ("rest", "restful", "rest api")),
    ("GraphQL", ("graphql",)),
    ("gRPC", ("grpc",)),
    ("Kafka", ("kafka",)),
    ("Spark", ("spark", "pyspark", "apache spark")),
    ("Hadoop", ("hadoop",)),
    ("Airflow", ("airflow", "apache airflow")),
    ("dbt", ("dbt",)),
    ("Snowflake", ("snowflake",)),
    ("BigQuery", ("bigquery", "big query")),
    ("Redshift", ("redshift",)),
    ("Pandas", ("pandas",)),
    ("NumPy", ("numpy",)),
    ("Machine Learning", ("machine learning", "ml")),
    ("Deep Learning", ("deep learning",)),
    ("NLP", ("nlp", "natural language processing")),
    ("Computer Vision", ("computer vision",)),
    ("PyTorch", ("pytorch", "torch")),
    ("TensorFlow", ("tensorflow",)),
    ("Scikit-learn", ("scikit-learn", "sklearn", "scikit learn")),
    ("Hugging Face", ("hugging face", "huggingface", "transformers")),
    ("LLM", ("llm", "llms", "large language model", "large language models")),
    ("RAG", ("rag", "retrieval augmented generation", "retrieval-augmented")),
    ("LangChain", ("langchain",)),
    ("ChromaDB", ("chromadb", "chroma")),
    ("OpenAI", ("openai", "gpt", "gpt-4", "gpt-4o")),
    ("Pydantic", ("pydantic",)),
    ("SQLModel", ("sqlmodel",)),
    ("SQLite", ("sqlite",)),
    ("C++", ("c++", "cpp")),
    ("C#", ("c#", "csharp")),
    (".NET", (".net", "dotnet", "asp.net")),
    ("Go", ("golang", "go lang", "go")),
    ("Rust", ("rust",)),
    ("Scala", ("scala",)),
    ("R", ("r language", "r programming", "tidyverse", "ggplot", "ggplot2")),
    ("Excel", ("excel", "vlookup", "pivot table")),
    ("Tableau", ("tableau",)),
    ("Power BI", ("power bi", "powerbi")),
    ("SparkSQL", ("sparksql", "spark sql")),
    ("Airbyte", ("airbyte",)),
    ("dbt Core", ("dbt core",)),
    ("SageMaker", ("sagemaker",)),
    ("Lambda", ("aws lambda", "lambda")),
    ("S3", ("s3", "amazon s3")),
    ("EC2", ("ec2",)),
    ("ECS", ("ecs", "fargate")),
    ("EKS", ("eks",)),
    ("Prometheus", ("prometheus",)),
    ("Grafana", ("grafana",)),
    ("Datadog", ("datadog",)),
    ("pytest", ("pytest",)),
    ("Selenium", ("selenium",)),
    ("Playwright", ("playwright",)),
    ("Jira", ("jira",)),
    ("Agile", ("agile", "scrum")),
    ("Microservices", ("microservices", "microservice")),
    ("System Design", ("system design",)),
    ("Data Engineering", ("data engineering", "data engineer")),
    ("ETL", ("etl", "elt")),
    ("Feature Engineering", ("feature engineering",)),
    ("MLOps", ("mlops",)),
    ("DevOps", ("devops",)),
    ("Security", ("oauth", "jwt", "sso", "owasp")),
    ("Kafka Streams", ("kafka streams",)),
    ("Celery", ("celery",)),
    ("RabbitMQ", ("rabbitmq",)),
    ("Nginx", ("nginx",)),
    ("Redis Cache", ("redis cache",)),
)

_SKILL_PATTERNS: List[Tuple[re.Pattern[str], str]] = []


def _init_skill_patterns() -> None:
    """Compile alias regexes once (longest aliases first)."""
    if _SKILL_PATTERNS:
        return
    items: List[Tuple[str, str]] = []
    for canonical, aliases in _SKILL_GROUPS:
        for alias in aliases:
            items.append((alias.strip().lower(), canonical))
    items.sort(key=lambda pair: len(pair[0]), reverse=True)
    for alias, canonical in items:
        escaped: str = re.escape(alias)
        pattern = re.compile(rf"(?<![a-z0-9_+#]){escaped}(?![a-z0-9_+#])", re.IGNORECASE)
        _SKILL_PATTERNS.append((pattern, canonical))


def extract_technical_skills(text: str) -> List[str]:
    """Return canonical technical skills mentioned in text (order-stable, unique)."""
    _init_skill_patterns()
    if not text:
        return []
    found: List[str] = []
    seen: set[str] = set()
    haystack: str = text.lower()
    for pattern, canonical in _SKILL_PATTERNS:
        if canonical in seen:
            continue
        if pattern.search(haystack) is not None:
            seen.add(canonical)
            found.append(canonical)
    return found


class EvaluationSchema(BaseModel):
    """Structured LLM output for a resume vs job-description match."""

    llm_score: float = Field(..., description="Match score from 1.0 to 10.0")
    strengths: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")

    @field_validator("llm_score")
    @classmethod
    def clamp_llm_score(cls, value: float) -> float:
        numeric: float = float(value)
        return max(1.0, min(10.0, numeric))

    @field_validator("strengths", "missing_skills", mode="before")
    @classmethod
    def coerce_string_lists(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value)]


def _load_embedder() -> Any:
    """Load the MiniLM bi-encoder once per process."""
    global _embedder
    if _embedder is not None:
        return _embedder
    try:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(BI_ENCODER_NAME)
        return _embedder
    except Exception as exc:
        logger.exception("Failed to load SentenceTransformer: %s", exc)
        raise RuntimeError("Embedding model could not be loaded.") from exc


def _load_cross_encoder() -> Any:
    """Load the MS MARCO Cross-Encoder once per process. Returns None if unavailable."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder_failed:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(CROSS_ENCODER_NAME)
        return _cross_encoder
    except Exception as exc:
        _cross_encoder_failed = True
        logger.warning("Cross-Encoder unavailable (%s); RRF scores will be used instead.", exc)
        return None


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two dense vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot: float = 0.0
    norm_a: float = 0.0
    norm_b: float = 0.0
    for a_val, b_val in zip(vec_a, vec_b):
        dot += a_val * b_val
        norm_a += a_val * a_val
        norm_b += b_val * b_val
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def _bm25_tokens(text: str) -> List[str]:
    tokens: List[str] = re.findall(r"[a-z0-9+#]+", (text or "").lower())
    return tokens if tokens else ["empty"]


def _tokenize(text: str) -> set[str]:
    tokens: List[str] = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}", text.lower())
    stopwords: set[str] = {
        "the", "and", "for", "with", "that", "this", "from", "have", "has",
        "was", "were", "are", "you", "your", "our", "their", "will", "can",
        "job", "role", "team", "work", "experience", "years", "using",
    }
    return {token for token in tokens if token not in stopwords and len(token) > 2}


def _lexical_overlap_score(resume_text: str, job_desc: str) -> float:
    """Jaccard overlap used when embeddings are unavailable."""
    resume_tokens: set[str] = _tokenize(resume_text)
    jd_tokens: set[str] = _tokenize(job_desc)
    if not resume_tokens or not jd_tokens:
        return 0.0
    intersection: int = len(resume_tokens & jd_tokens)
    union: int = len(resume_tokens | jd_tokens)
    return max(0.0, min(100.0, (intersection / union) * 100.0))


def compute_vector_score(resume_text: str, job_desc: str) -> float:
    """Return cosine similarity between resume and JD as a 0-100 percentage."""
    try:
        embedder = _load_embedder()
        embeddings = embedder.encode(
            [resume_text[:8000], job_desc[:8000]],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        score: float = cosine_similarity(embeddings[0].tolist(), embeddings[1].tolist())
        return max(0.0, min(100.0, score * 100.0))
    except Exception as exc:
        logger.exception("Vector scoring failed, using lexical fallback: %s", exc)
        return _lexical_overlap_score(resume_text, job_desc)


def _compute_vector_scores(resume_texts: Sequence[str], job_desc: str) -> List[float]:
    """Batch bi-encoder cosine similarities (0-100)."""
    if not resume_texts:
        return []
    try:
        embedder = _load_embedder()
        payload: List[str] = [job_desc[:8000]] + [text[:8000] for text in resume_texts]
        embeddings = embedder.encode(payload, convert_to_numpy=True, show_progress_bar=False)
        jd_vec: List[float] = embeddings[0].tolist()
        scores: List[float] = []
        for index in range(1, len(embeddings)):
            cosine: float = cosine_similarity(embeddings[index].tolist(), jd_vec)
            scores.append(max(0.0, min(100.0, cosine * 100.0)))
        return scores
    except Exception as exc:
        logger.warning("Batch vector scoring failed (%s); using lexical overlap.", exc)
        return [_lexical_overlap_score(text, job_desc) for text in resume_texts]


def _compute_bm25_scores(resume_texts: Sequence[str], job_desc: str) -> List[float]:
    """BM25Okapi lexical scores for each resume against the job description."""
    try:
        from rank_bm25 import BM25Okapi

        corpus: List[List[str]] = [_bm25_tokens(text) for text in resume_texts]
        query: List[str] = _bm25_tokens(job_desc)
        bm25 = BM25Okapi(corpus)
        raw_scores = bm25.get_scores(query)
        return [float(score) for score in raw_scores]
    except Exception as exc:
        logger.warning("BM25 scoring failed (%s); using token overlap.", exc)
        return [_lexical_overlap_score(text, job_desc) for text in resume_texts]


def _ranks_desc(scores: Sequence[float]) -> List[int]:
    """1-indexed ranks; rank 1 is the highest score. Ties keep earlier documents first."""
    order: List[int] = sorted(
        range(len(scores)),
        key=lambda idx: (-float(scores[idx]), idx),
    )
    ranks: List[int] = [0] * len(scores)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _rrf_scores(bm25_ranks: Sequence[int], vector_ranks: Sequence[int]) -> List[float]:
    fused: List[float] = []
    for bm25_rank, vector_rank in zip(bm25_ranks, vector_ranks):
        fused.append((1.0 / (RRF_K + int(bm25_rank))) + (1.0 / (RRF_K + int(vector_rank))))
    return fused


def _sigmoid(value: float) -> float:
    clipped: float = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-clipped))


def _normalize_rrf(rrf_values: Sequence[float]) -> List[float]:
    ceiling: float = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 1)
    return [max(0.0, min(100.0, (float(value) / ceiling) * 100.0)) for value in rrf_values]


def _cross_encoder_percentages(
    resume_texts: Sequence[str],
    job_desc: str,
    fallback_percentages: Sequence[float],
) -> Tuple[List[float], str]:
    """Return (0-100 scores, scorer_name). Falls back to RRF percentages on failure."""
    reranker = _load_cross_encoder()
    if reranker is None:
        return [float(item) for item in fallback_percentages], "rrf"
    try:
        pairs: List[Tuple[str, str]] = [
            (job_desc[:1500], text[:1500]) for text in resume_texts
        ]
        logits = reranker.predict(pairs, show_progress_bar=False)
        percentages: List[float] = []
        iterable: Iterable[Any] = logits if hasattr(logits, "__iter__") and not isinstance(logits, (str, bytes)) else [logits]
        for logit in iterable:
            percentages.append(max(0.0, min(100.0, _sigmoid(float(logit)) * 100.0)))
        if len(percentages) != len(resume_texts):
            raise RuntimeError("Cross-Encoder returned an unexpected number of scores.")
        return percentages, "cross-encoder"
    except Exception as exc:
        logger.warning("Cross-Encoder scoring failed (%s); using RRF fallback.", exc)
        return [float(item) for item in fallback_percentages], "rrf"


def _skill_evaluation(
    resume_text: str,
    job_desc: str,
    match_percentage: float,
    vector_percentage: float,
    bm25_rank: int,
    vector_rank: int,
    rrf_score: float,
    scorer_name: str,
) -> EvaluationSchema:
    """Catalog-based strengths/gaps for local fallback (no generic stop-word tags)."""
    resume_skills: List[str] = extract_technical_skills(resume_text)
    jd_skills: List[str] = extract_technical_skills(job_desc)
    resume_set: set[str] = set(resume_skills)
    jd_set: set[str] = set(jd_skills)
    matched: List[str] = [skill for skill in jd_skills if skill in resume_set]
    missing: List[str] = [skill for skill in jd_skills if skill not in resume_set]
    extra: List[str] = [skill for skill in resume_skills if skill not in jd_set]

    strengths: List[str]
    if matched:
        strengths = matched[:10]
    elif extra:
        strengths = extra[:10]
    else:
        strengths = ["No catalogued technical skills detected in the resume"]

    gaps: List[str] = missing[:10] if missing else ["No major technical skill gaps detected"]
    required: int = max(len(jd_set), 1)
    coverage: float = (len(matched) / required) * 100.0 if jd_set else 0.0
    llm_score: float = max(1.0, min(10.0, match_percentage / 10.0))
    matched_label: str = ", ".join(matched[:8]) if matched else "none"
    missing_label: str = ", ".join(missing[:8]) if missing else "none"
    reasoning: str = (
        f"Local hybrid search ({scorer_name}). "
        f"BM25 rank {bm25_rank}, vector rank {vector_rank}, "
        f"RRF {rrf_score:.4f}. Technical coverage {coverage:.1f}%. "
        f"Matched: {matched_label}. Gaps: {missing_label}. "
        f"Bi-encoder cosine {vector_percentage:.1f}%."
    )
    return EvaluationSchema(
        llm_score=round(llm_score, 2),
        strengths=strengths,
        missing_skills=gaps,
        reasoning=reasoning,
    )


def _call_openai(resume_text: str, job_desc: str) -> Tuple[EvaluationSchema, int]:
    """Call OpenAI chat completions and parse EvaluationSchema JSON. Returns (schema, tokens)."""
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, timeout=45.0)
    model_name: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    prompt: str = (
        "You are a senior technical recruiter. Score how well the resume matches the job. "
        "Return ONLY valid JSON with keys: llm_score (1.0-10.0), strengths (string array), "
        "missing_skills (string array), reasoning (string). "
        "Strengths and missing_skills must be concrete technical skills, never generic words.\n\n"
        f"JOB DESCRIPTION:\n{job_desc[:6000]}\n\n"
        f"RESUME:\n{resume_text[:8000]}"
    )
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return strict JSON matching the requested schema."},
            {"role": "user", "content": prompt},
        ],
    )
    content: str = (response.choices[0].message.content or "").strip()
    tokens: int = 0
    if response.usage is not None:
        tokens = int(response.usage.total_tokens or 0)
    parsed: Dict[str, Any] = json.loads(content)
    schema = EvaluationSchema.model_validate(parsed)
    return schema, tokens


def _assemble_payload(
    resume_text: str,
    job_desc: str,
    vector_percentage: float,
    match_percentage: float,
    bm25_rank: int,
    vector_rank: int,
    rrf_score: float,
    scorer_name: str,
    started: float,
    use_openai: bool,
) -> Dict[str, Any]:
    """Build the FastAPI/Streamlit evaluation dict without breaking existing keys."""
    tokens_used: int = 0
    source: str = "hybrid-local"
    llm_result: EvaluationSchema = _skill_evaluation(
        resume_text,
        job_desc,
        match_percentage,
        vector_percentage,
        bm25_rank,
        vector_rank,
        rrf_score,
        scorer_name,
    )

    if use_openai:
        try:
            llm_result, tokens_used = _call_openai(resume_text, job_desc)
            source = "openai+hybrid"
        except Exception as exc:
            logger.warning("OpenAI evaluation unavailable (%s); keeping local hybrid skills.", exc)
            source = "hybrid-local"

    llm_score: float
    if source.startswith("openai"):
        llm_score = float(llm_result.llm_score)
    else:
        llm_score = max(0.0, min(10.0, match_percentage / 10.0))

    token_cost: float = (tokens_used / 1000.0) * COST_PER_1K_TOKENS
    latency_seconds: float = time.perf_counter() - started
    final_score: float = max(0.0, min(100.0, match_percentage))
    return {
        "vector_score": round(vector_percentage, 2),
        "vector_percentage": round(vector_percentage, 2),
        "llm_score": round(llm_score, 2),
        "final_score": round(final_score, 2),
        "match_percentage": round(final_score, 2),
        "bm25_rank": bm25_rank,
        "vector_rank": vector_rank,
        "rrf_score": round(rrf_score, 6),
        "strengths": llm_result.strengths,
        "missing_skills": llm_result.missing_skills,
        "reasoning": llm_result.reasoning,
        "token_cost": round(token_cost, 6),
        "tokens_used": tokens_used,
        "latency_seconds": round(latency_seconds, 4),
        "source": source,
    }


def evaluate_matches(resume_texts: Sequence[str], job_desc: str) -> List[Dict[str, Any]]:
    """Hybrid BM25 + dense retrieval + RRF + Cross-Encoder scoring for a candidate set."""
    started: float = time.perf_counter()
    if not job_desc.strip():
        raise ValueError("Job description is empty.")
    cleaned: List[str] = []
    for text in resume_texts:
        if not str(text).strip():
            raise ValueError("Resume text is empty.")
        cleaned.append(str(text))
    if not cleaned:
        return []

    vector_percentages: List[float] = _compute_vector_scores(cleaned, job_desc)
    bm25_raw: List[float] = _compute_bm25_scores(cleaned, job_desc)
    bm25_ranks: List[int] = _ranks_desc(bm25_raw)
    vector_ranks: List[int] = _ranks_desc(vector_percentages)
    rrf_values: List[float] = _rrf_scores(bm25_ranks, vector_ranks)
    rrf_percentages: List[float] = _normalize_rrf(rrf_values)
    match_percentages, scorer_name = _cross_encoder_percentages(cleaned, job_desc, rrf_percentages)
    use_openai: bool = bool((os.getenv("OPENAI_API_KEY") or "").strip())

    results: List[Dict[str, Any]] = []
    for index, resume_text in enumerate(cleaned):
        results.append(
            _assemble_payload(
                resume_text=resume_text,
                job_desc=job_desc,
                vector_percentage=vector_percentages[index],
                match_percentage=match_percentages[index],
                bm25_rank=bm25_ranks[index],
                vector_rank=vector_ranks[index],
                rrf_score=rrf_values[index],
                scorer_name=scorer_name,
                started=started,
                use_openai=use_openai,
            )
        )
    return results


def evaluate_match(resume_text: str, job_desc: str) -> Dict[str, Any]:
    """Score a single resume with the same hybrid pipeline used for batches."""
    batch: List[Dict[str, Any]] = evaluate_matches([resume_text], job_desc)
    if not batch:
        raise ValueError("Resume text is empty.")
    return batch[0]


def warm_models() -> None:
    """Best-effort cache of bi-encoder and cross-encoder at process start."""
    try:
        _load_embedder()
    except Exception as exc:
        logger.warning("Bi-encoder warmup skipped: %s", exc)
    try:
        _load_cross_encoder()
    except Exception as exc:
        logger.warning("Cross-encoder warmup skipped: %s", exc)
