"""PDF text extraction and regex-based PII anonymization."""

from __future__ import annotations

import io
import logging
import re
from typing import Dict, List, Tuple

import pdfplumber

logger = logging.getLogger(__name__)

EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)
PHONE_PATTERN: re.Pattern[str] = re.compile(
    r"(?:(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4})"
)
URL_PATTERN: re.Pattern[str] = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+",
    re.IGNORECASE,
)
LINKEDIN_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:linkedin\.com/in/|linkedin\.com/pub/)[^\s<>\"']+",
    re.IGNORECASE,
)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract concatenated page text from a PDF byte payload."""
    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty.")

    pages_text: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                raise ValueError("PDF contains no pages.")
            for page in pdf.pages:
                page_text: str = page.extract_text() or ""
                pages_text.append(page_text.strip())
    except ValueError:
        raise
    except Exception as exc:
        logger.exception("PDF extraction failed: %s", exc)
        raise ValueError(f"Unable to parse PDF: {exc}") from exc

    combined: str = "\n\n".join(chunk for chunk in pages_text if chunk)
    if not combined.strip():
        raise ValueError("No extractable text found in PDF (it may be image-only).")
    return combined.strip()


def anonymize_text(text: str) -> str:
    """Strip emails, phone numbers, and URLs from resume text."""
    if not text:
        return ""
    try:
        redacted: str = EMAIL_PATTERN.sub("[EMAIL_REDACTED]", text)
        redacted = LINKEDIN_PATTERN.sub("[PROFILE_REDACTED]", redacted)
        redacted = URL_PATTERN.sub("[URL_REDACTED]", redacted)
        redacted = PHONE_PATTERN.sub("[PHONE_REDACTED]", redacted)
        redacted = re.sub(r"\n{3,}", "\n\n", redacted)
        return redacted.strip()
    except Exception as exc:
        logger.exception("Anonymization failed: %s", exc)
        return text


def parse_resume(pdf_bytes: bytes, anonymize: bool = True) -> Tuple[str, str, Dict[str, object]]:
    """Extract raw and optionally anonymized text plus lightweight metadata."""
    raw_text: str = extract_text_from_pdf(pdf_bytes)
    anonymized: str = anonymize_text(raw_text) if anonymize else raw_text
    metadata: Dict[str, object] = {
        "word_count": len(raw_text.split()),
        "char_count": len(raw_text),
        "anonymized": anonymize,
        "pii_email_count": len(EMAIL_PATTERN.findall(raw_text)),
        "pii_url_count": len(URL_PATTERN.findall(raw_text)),
        "pii_phone_count": len(PHONE_PATTERN.findall(raw_text)),
    }
    return raw_text, anonymized, metadata
