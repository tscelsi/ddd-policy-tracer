"""Application service orchestration for entities bounded context."""

from __future__ import annotations

import logging
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

_LOGGER = logging.getLogger(__name__)


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
        _LOGGER.info("Starting entity extraction for chunk_id=%s", chunk_id)
        try:
            chunk = self.chunk_repository.get_chunk(chunk_id=chunk_id)
        except Exception as exc:
            _LOGGER.exception(
                "Failed loading chunk for extraction chunk_id=%s error=%s",
                chunk_id,
                exc,
            )
            return self._publish_failure(
                chunk_id=chunk_id,
                processed_sentences=0,
                error_message=f"failed loading chunk: {exc}",
            )
        if chunk is None:
            _LOGGER.warning("Chunk not found for extraction chunk_id=%s", chunk_id)
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
            _LOGGER.exception(
                "Entity extraction failed for chunk_id=%s with error=%s",
                chunk_id,
                exc,
            )
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
        _LOGGER.info(
            "Completed entity extraction for chunk_id=%s entities_extracted=%s",
            chunk_id,
            entities_extracted,
        )
        return report

    def extract_entities_for_all_chunks(self) -> list[EntityExtractionReport]:
        """Extract and persist entities for every chunk in repository state."""
        try:
            chunk_ids = self.chunk_repository.list_chunk_ids()
        except Exception as exc:
            _LOGGER.exception("Failed listing chunks for bulk extraction error=%s", exc)
            return [
                self._publish_failure(
                    chunk_id="<chunk-list>",
                    processed_sentences=0,
                    error_message=f"failed listing chunks: {exc}",
                ),
            ]
        _LOGGER.info("Starting bulk entity extraction for chunk_count=%s", len(chunk_ids))
        chunks_by_id = {}
        for chunk_id in chunk_ids:
            try:
                chunks_by_id[chunk_id] = self.chunk_repository.get_chunk(chunk_id=chunk_id)
            except Exception as exc:
                _LOGGER.exception(
                    "Failed loading chunk during bulk extraction chunk_id=%s error=%s",
                    chunk_id,
                    exc,
                )
                chunks_by_id[chunk_id] = None
        chunks = [chunk for chunk in chunks_by_id.values() if chunk is not None]

        try:
            entities = self.extractor.extract_many(chunks=chunks)
        except Exception as exc:
            _LOGGER.exception(
                "Bulk entity extraction failed for chunk_count=%s with error=%s",
                len(chunk_ids),
                exc,
            )
            return [
                self._publish_failure(
                    chunk_id=chunk_id,
                    processed_sentences=0,
                    error_message=str(exc),
                )
                for chunk_id in chunk_ids
            ]

        existing_counts_by_chunk = {
            chunk_id: len(self.entity_repository.list_entities(chunk_id=chunk_id))
            for chunk_id in chunk_ids
        }
        self.entity_repository.add_entities(entities)

        entities_by_chunk_id: dict[str, list[EntityMention]] = {
            chunk.chunk_id: []
            for chunk in chunks
        }
        for entity in entities:
            chunk_entities = entities_by_chunk_id.get(entity.chunk_id)
            if chunk_entities is None:
                continue
            chunk_entities.append(entity)

        reports: list[EntityExtractionReport] = []
        for chunk_id in chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                _LOGGER.warning("Chunk missing during bulk extraction chunk_id=%s", chunk_id)
                reports.append(
                    self._publish_failure(
                        chunk_id=chunk_id,
                        processed_sentences=0,
                        error_message="chunk not found",
                    ),
                )
                continue

            chunk_entities = entities_by_chunk_id.get(chunk_id, [])
            entities_extracted = (
                len(self.entity_repository.list_entities(chunk_id=chunk_id))
                - existing_counts_by_chunk.get(chunk_id, 0)
            )
            processed_sentences = self.extractor.count_processed_sentences(chunk=chunk)

            report = EntityExtractionReport(
                chunk_id=chunk_id,
                status="completed",
                entities_extracted=entities_extracted,
                processed_sentences=processed_sentences,
                entities_by_type=self._count_entities_by_type(chunk_entities),
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
            reports.append(report)
        _LOGGER.info(
            "Completed bulk entity extraction for chunk_count=%s failed_count=%s",
            len(reports),
            sum(1 for report in reports if report.status == "failed"),
        )
        return reports

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
