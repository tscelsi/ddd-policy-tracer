"""Application service orchestration for entities bounded context."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .models import EntityExtractionReport, EntityMention, EntityType
from .ports import (
    ChunkRepository,
    EntityExtractor,
    EntityRepository,
    EventPublisher,
)


@dataclass(frozen=True)
class EntitiesService:
    """Coordinate chunk loading, extraction, persistence, and eventing."""

    chunk_repository: ChunkRepository
    entity_repository: EntityRepository
    extractor: EntityExtractor
    event_publisher: EventPublisher

    _ENTITY_TYPES: tuple[EntityType, ...] = (
        "POLICY",
        "ORG",
        "PERSON",
        "JURISDICTION",
        "PROGRAM",
    )

    def extract_entities_for_chunk(self, *, chunk_id: str) -> EntityExtractionReport:
        """Extract and persist entities for one chunk identity."""
        chunk = self.chunk_repository.get_chunk(chunk_id=chunk_id)
        if chunk is None:
            return self._publish_failure(
                chunk_id=chunk_id,
                processed_sentences=0,
                error_message="chunk not found",
            )

        try:
            processed_sentences = self.extractor.count_processed_sentences(chunk=chunk)
            entities = self.extractor.extract(chunk=chunk)
            entities_extracted = self.entity_repository.add_entities(entities)
        except Exception as exc:
            return self._publish_failure(
                chunk_id=chunk_id,
                processed_sentences=0,
                error_message=str(exc),
            )

        report = EntityExtractionReport(
            chunk_id=chunk_id,
            status="completed",
            entities_extracted=entities_extracted,
            processed_sentences=processed_sentences,
            entities_by_type=self._count_entities_by_type(entities),
            error_message=None,
        )
        self.event_publisher.publish(
            {
                "topic": "entities.extraction.status",
                "chunk_id": chunk_id,
                "status": report.status,
                "entities_extracted": report.entities_extracted,
                "processed_sentences": report.processed_sentences,
                "entities_by_type": report.entities_by_type,
            },
        )
        return report

    def _publish_failure(
        self,
        *,
        chunk_id: str,
        processed_sentences: int,
        error_message: str,
    ) -> EntityExtractionReport:
        """Publish a failed status event and return a failure report."""
        report = EntityExtractionReport(
            chunk_id=chunk_id,
            status="failed",
            entities_extracted=0,
            processed_sentences=processed_sentences,
            entities_by_type=self._count_entities_by_type(()),
            error_message=error_message,
        )
        self.event_publisher.publish(
            {
                "topic": "entities.extraction.status",
                "chunk_id": chunk_id,
                "status": report.status,
                "entities_extracted": report.entities_extracted,
                "processed_sentences": report.processed_sentences,
                "entities_by_type": report.entities_by_type,
            },
        )
        return report

    def _count_entities_by_type(
        self,
        entities: Sequence[EntityMention],
    ) -> dict[EntityType, int]:
        """Count extracted entities by strict v1 entity type values."""
        counts = Counter(entity.entity_type for entity in entities)
        return {entity_type: counts.get(entity_type, 0) for entity_type in self._ENTITY_TYPES}
