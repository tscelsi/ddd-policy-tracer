"""Ports for entities bounded context orchestration dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ddd_policy_tracer.analysis.chunking_models import DocumentChunk

from .models import EntityMention


@dataclass(frozen=True)
class ExtractionMetrics:
    """Represent extractor metrics used by service reporting and events."""

    processed_sentences: int


class ChunkRepository(Protocol):
    """Load chunks by identity for entity extraction workflows."""

    def get_chunk(self, *, chunk_id: str) -> DocumentChunk | None:
        """Return one chunk for extraction or none when missing."""


class EntityRepository(Protocol):
    """Persist extracted entity mentions."""

    def add_entities(self, entities: list[EntityMention]) -> int:
        """Persist entity mentions and return inserted mention count."""

    def list_entities(self, *, chunk_id: str | None = None) -> list[EntityMention]:
        """Load persisted entity mentions, optionally by chunk identity."""


class EntityExtractor(Protocol):
    """Extract entity mentions from one source chunk."""

    def extract(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Return extracted entity mentions for one chunk."""

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Return number of processed sentences for one chunk."""


class EventPublisher(Protocol):
    """Publish compact orchestration status events."""

    def publish(self, event: dict[str, object]) -> None:
        """Publish one event for downstream processing."""
