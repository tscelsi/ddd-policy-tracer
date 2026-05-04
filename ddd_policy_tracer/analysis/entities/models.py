"""Domain records for entity extraction service orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EntityType = Literal["POLICY", "ORG", "PERSON", "JURISDICTION", "PROGRAM"]


@dataclass(frozen=True)
class EntityMention:
    """Represent one extracted entity mention for persistence and review."""

    entity_id: str
    chunk_id: str
    source_id: str
    source_document_id: str
    document_checksum: str
    start_char: int
    end_char: int
    mention_text: str
    normalized_mention_text: str
    entity_type: EntityType
    confidence: float
    extractor_version: str
    canonical_entity_key: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class EntityExtractionReport:
    """Capture one chunk extraction outcome for synchronous callers."""

    chunk_id: str
    status: Literal["completed", "failed"]
    entities_extracted: int
    processed_sentences: int
    entities_by_type: dict[EntityType, int]
    error_message: str | None
