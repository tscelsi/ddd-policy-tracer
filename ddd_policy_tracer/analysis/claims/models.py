"""Domain records for claim extraction service orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ClaimCandidate:
    """Represent one extracted claim candidate for persistence and review."""

    evidence_text: str


@dataclass(frozen=True)
class ClaimExtractionReport:
    """Capture one chunk extraction outcome for synchronous callers."""

    chunk_id: str
    status: Literal["completed", "failed"]
    claims_extracted: int
    processed_sentences: int
    error_message: str | None
