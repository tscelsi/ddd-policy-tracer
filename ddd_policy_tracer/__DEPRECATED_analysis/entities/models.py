"""Domain records for entity extraction service orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EntityType = Literal["POLICY", "ORG", "PERSON", "JURISDICTION", "PROGRAM"]

CATALOG_ELIGIBLE_ENTITY_TYPES: tuple[EntityType, ...] = (
    "ORG",
    "PERSON",
    "JURISDICTION",
)


@dataclass
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


#### CATALOG MODELS ####

ReviewState = Literal["pending", "approved", "rejected", "merged"]


@dataclass
class CatalogCandidate:
    """Represent one extracted entity mention promoted for catalog review."""

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
    review_state: ReviewState
    canonical_entity_key: str | None = None
    metadata: dict[str, object] | None = None

    def set_review_state(
        self,
        new_state: ReviewState,
    ) -> None:
        """Return a new candidate instance with updated review state."""
        self.review_state = new_state

    def promote_by_creation(
        self,
        key: str,
        name: str,
        entity_type: EntityType,
        description: str | None = None,
    ) -> CatalogEntity:
        """Return a new catalog entity instance by promoting this candidate."""
        self.set_review_state("approved")
        return CatalogEntity(
            canonical_entity_key=key,
            canonical_name=name,
            entity_type=entity_type,
            description=description,
        )

    @classmethod
    def from_entity_mention(
        cls,
        mention: EntityMention,
        review_state: ReviewState = "pending",
    ) -> CatalogCandidate:
        """Return a new candidate instance by promoting one entity mention."""
        return cls(
            entity_id=mention.entity_id,
            chunk_id=mention.chunk_id,
            source_id=mention.source_id,
            source_document_id=mention.source_document_id,
            document_checksum=mention.document_checksum,
            start_char=mention.start_char,
            end_char=mention.end_char,
            mention_text=mention.mention_text,
            normalized_mention_text=mention.normalized_mention_text,
            entity_type=mention.entity_type,
            confidence=mention.confidence,
            extractor_version=mention.extractor_version,
            canonical_entity_key=mention.canonical_entity_key or "",
            metadata=mention.metadata,
            review_state=review_state,
        )


@dataclass
class CatalogEntity:
    """Represent one canonical entity in the catalog."""

    canonical_entity_key: str
    canonical_name: str
    entity_type: EntityType
    description: str | None = None
    contexts: list[str] | None = None
    aliases: list[str] | None = None
    embedding: list[float] | None = None

    def add_aliases(self, new_aliases: list[str]) -> None:
        """Add new aliases to this catalog entity."""
        if self.aliases is None:
            self.aliases = []
        self.aliases.extend(new_aliases)
        self.aliases = list(set(self.aliases))  # Ensure uniqueness

    def add_contexts(self, ctxs: list[str]) -> None:
        """Append new contexts to this catalog entity."""
        if self.contexts is None:
            self.contexts = []
        self.contexts.extend(ctxs)

    def update_description(self, new_description: str) -> None:
        """Update the description of this catalog entity."""
        self.description = new_description

    def update_embedding(self, new_embedding: list[float]) -> None:
        """Update the embedding of this catalog entity."""
        self.embedding = new_embedding

    def __str__(self) -> str:
        """Return a human-readable string representation of this entity."""
        return (
            f"{self.canonical_name}\n"
            + f"Type: {self.entity_type}\n"
            + (f"Description: {self.description}\n" if self.description else "")
            + (f"Aliases: {', '.join(self.aliases)}\n" if self.aliases else "")
            + (f"Example mentions: {', '.join(self.contexts)}\n" if self.contexts else "")
        )

    def __repr__(self) -> str:
        """Return a concise string representation of this entity."""
        return f"CatalogEntity(key={self.canonical_entity_key}, name={self.canonical_name}, type={self.entity_type})"
