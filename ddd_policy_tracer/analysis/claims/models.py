"""Domain records for claim extraction service orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ClaimCandidate:
    """Represent one extracted claim candidate for persistence and review."""

    claim_id: str
    chunk_id: str
    source_id: str
    source_document_id: str
    document_checksum: str
    start_char: int
    end_char: int
    evidence_text: str
    normalized_claim_text: str
    confidence: float
    claim_type: str | None
    extractor_version: str


@dataclass(frozen=True)
class ClaimExtractionReport:
    """Capture one chunk extraction outcome for synchronous callers."""

    chunk_id: str
    status: Literal["completed", "failed"]
    claims_extracted: int
    processed_sentences: int
    error_message: str | None
