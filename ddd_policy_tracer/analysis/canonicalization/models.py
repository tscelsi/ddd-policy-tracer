"""Domain records for canonicalization stage outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CanonicalizationMetadata:
    """Record canonicalization run metadata embedded in each row."""

    stage: Literal["entity", "claim"]
    canonicalizer_version: str
    generated_at: str
    input_artifact_hash: str


@dataclass(frozen=True)
class CanonicalEntityRecord:
    """Represent one canonicalized entity row with deterministic identity."""

    entity_id: str
    chunk_id: str
    source_id: str
    source_document_id: str
    document_checksum: str
    start_char: int
    end_char: int
    mention_text: str
    normalized_mention_text: str
    entity_type: str
    confidence: float
    extractor_version: str
    canonical_name: str
    canonical_entity_key: str
    canonicalization: CanonicalizationMetadata


@dataclass(frozen=True)
class LinkedEntityEvidence:
    """Capture link evidence details for one claim-linked entity."""

    claim_span: tuple[int, int]
    entity_span: tuple[int, int]
    matched_text: str | None


@dataclass(frozen=True)
class LinkedEntityRecord:
    """Represent one canonical linked entity attached to a claim row."""

    canonical_entity_key: str
    entity_type: str
    canonical_name: str
    link_method: Literal["span_overlap", "text_match_fallback"]
    entity_id: str
    evidence: LinkedEntityEvidence | None


@dataclass(frozen=True)
class CanonicalClaimRecord:
    """Represent one canonicalized claim row with linked entities."""

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
    linked_entities: list[LinkedEntityRecord]
    canonicalization: CanonicalizationMetadata


@dataclass(frozen=True)
class CanonicalizationReport:
    """Capture one canonicalization run result for CLI/reporting."""

    stage: Literal["entity", "claim"]
    input_rows: int
    output_rows: int
    canonicalizer_version: str
    generated_at: str
