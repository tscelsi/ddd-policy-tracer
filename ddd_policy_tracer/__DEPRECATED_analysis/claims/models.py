"""Domain records for claim extraction service orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import spacy


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
    topics: list[str] | None = None
    verbs: list[str] | None = None
    adjectives: list[str] | None = None
    nouns: list[str] | None = None

    def enrich(self, nlp: spacy.language.Language) -> ClaimCandidate:
        """Enrich claim candidate with extracted topics, verbs, adjectives, and nouns."""
        doc = nlp(self.evidence_text)
        verbs = list({token.lemma_ for token in doc if token.pos_ == "VERB"})
        adjectives = list({token.lemma_ for token in doc if token.pos_ == "ADJ"})
        nouns = list({token.lemma_ for token in doc if token.pos_ == "NOUN"})
        return ClaimCandidate(
            claim_id=self.claim_id,
            chunk_id=self.chunk_id,
            source_id=self.source_id,
            source_document_id=self.source_document_id,
            document_checksum=self.document_checksum,
            start_char=self.start_char,
            end_char=self.end_char,
            evidence_text=self.evidence_text,
            normalized_claim_text=self.normalized_claim_text,
            confidence=self.confidence,
            claim_type=self.claim_type,
            extractor_version=self.extractor_version,
            verbs=verbs,
            adjectives=adjectives,
            nouns=nouns,
        )


@dataclass(frozen=True)
class ClaimExtractionReport:
    """Capture one chunk extraction outcome for synchronous callers."""

    chunk_id: str
    status: Literal["completed", "failed"]
    claims_extracted: int
    processed_sentences: int
    error_message: str | None
