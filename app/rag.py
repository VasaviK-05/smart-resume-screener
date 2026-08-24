"""ChromaDB vector store for recruiter natural-language queries over resumes."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

logger = logging.getLogger(__name__)

CHROMA_DIR: Path = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME: str = "resume_chunks"
CHUNK_SIZE: int = 700
CHUNK_OVERLAP: int = 80
DEFAULT_TOP_K: int = 5

_client: Any = None
_collection: Any = None


def _get_collection() -> Any:
    """Return a persistent Chroma collection, creating it if needed."""
    global _client, _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        return _collection
    except Exception as exc:
        logger.exception("Failed to initialize ChromaDB: %s", exc)
        raise RuntimeError("ChromaDB could not be initialized.") from exc


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split resume text into overlapping character windows on paragraph boundaries."""
    cleaned: str = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []
    paragraphs: List[str] = [p.strip() for p in re.split(r"\n{2,}", text.strip()) if p.strip()]
    source_units: List[str] = paragraphs if paragraphs else [cleaned]
    chunks: List[str] = []
    buffer: str = ""
    for unit in source_units:
        candidate: str = f"{buffer} {unit}".strip() if buffer else unit
        if len(candidate) <= chunk_size:
            buffer = candidate
            continue
        if buffer:
            chunks.append(buffer)
        if len(unit) <= chunk_size:
            buffer = unit
        else:
            start: int = 0
            while start < len(unit):
                end: int = min(start + chunk_size, len(unit))
                chunks.append(unit[start:end].strip())
                start = max(end - overlap, start + 1)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return [chunk for chunk in chunks if chunk]


def index_resumes(candidates_list: Sequence[Any]) -> int:
    """Chunk and upsert candidate resumes. Returns number of chunks indexed."""
    try:
        collection = _get_collection()
    except Exception as exc:
        logger.exception("index_resumes aborted: %s", exc)
        return 0

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for candidate in candidates_list:
        try:
            candidate_id: int
            filename: str
            anonymized: str
            if isinstance(candidate, dict):
                candidate_id = int(candidate.get("id") or 0)
                filename = str(candidate.get("filename") or "unknown")
                anonymized = str(candidate.get("anonymized_text") or candidate.get("raw_text") or "")
            else:
                candidate_id = int(getattr(candidate, "id", 0) or 0)
                filename = str(getattr(candidate, "filename", "") or "unknown")
                anonymized = str(
                    getattr(candidate, "anonymized_text", "")
                    or getattr(candidate, "raw_text", "")
                    or ""
                )
        except Exception:
            continue

        if not candidate_id or not anonymized.strip():
            continue

        for index, chunk in enumerate(chunk_text(anonymized)):
            ids.append(f"cand-{candidate_id}-chunk-{index}")
            documents.append(chunk)
            metadatas.append(
                {
                    "candidate_id": int(candidate_id),
                    "filename": filename,
                    "chunk_index": index,
                }
            )

    if not ids:
        return 0

    try:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("Indexed %s resume chunks", len(ids))
        return len(ids)
    except Exception as exc:
        logger.exception("Chroma upsert failed: %s", exc)
        return 0


def query_resumes(query_text: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Return top-k resume snippets with candidate references for a recruiter query."""
    if not query_text or not query_text.strip():
        return []

    try:
        collection = _get_collection()
        count: int = collection.count()
        if count == 0:
            return []
        n_results: int = max(1, min(top_k, count))
        raw = collection.query(
            query_texts=[query_text.strip()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.exception("Chroma query failed: %s", exc)
        return []

    documents: List[str] = (raw.get("documents") or [[]])[0]
    metadatas: List[Dict[str, Any]] = (raw.get("metadatas") or [[]])[0]
    distances: List[float] = (raw.get("distances") or [[]])[0]
    results: List[Dict[str, Any]] = []

    for doc, meta, distance in zip(documents, metadatas, distances):
        similarity: float = max(0.0, min(1.0, 1.0 - float(distance)))
        results.append(
            {
                "candidate_id": int(meta.get("candidate_id", 0)),
                "filename": str(meta.get("filename", "unknown")),
                "chunk_index": int(meta.get("chunk_index", 0)),
                "snippet": doc,
                "similarity": round(similarity * 100.0, 2),
            }
        )
    return results
