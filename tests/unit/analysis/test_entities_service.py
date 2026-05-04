"""Unit tests for entities service tracer-bullet orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from ddd_policy_tracer.analysis.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.entities import EntitiesService, EntityMention


@dataclass
class InMemoryChunkRepository:
    """Provide predictable chunk lookup by chunk identifier."""

    chunks: dict[str, DocumentChunk]

    def get_chunk(self, *, chunk_id: str) -> DocumentChunk | None:
        """Return one chunk object when present in memory."""
        return self.chunks.get(chunk_id)


@dataclass
class InMemoryEntityRepository:
    """Collect persisted entities in memory for service assertions."""

    persisted: list[EntityMention]

    def add_entities(self, entities: list[EntityMention]) -> int:
        """Persist entity mentions and report inserted count."""
        self.persisted.extend(entities)
        return len(entities)


class FixedExtractor:
    """Return deterministic entities and sentence counts for one chunk."""

    def extract(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Return fixed entity mentions for any chunk."""
        return [
            EntityMention(
                entity_id="entity_policy_1",
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                source_document_id=chunk.source_document_id,
                document_checksum=chunk.document_checksum,
                start_char=0,
                end_char=10,
                mention_text="Climate Act",
                normalized_mention_text="Climate Act",
                entity_type="POLICY",
                confidence=0.95,
                extractor_version="rules-v1",
            ),
            EntityMention(
                entity_id="entity_org_1",
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                source_document_id=chunk.source_document_id,
                document_checksum=chunk.document_checksum,
                start_char=12,
                end_char=31,
                mention_text="Australia Institute",
                normalized_mention_text="Australia Institute",
                entity_type="ORG",
                confidence=0.91,
                extractor_version="rules-v1",
            ),
        ]

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Return deterministic sentence count for report output."""
        _ = chunk
        return 2


@dataclass
class RecordingPublisher:
    """Record published events in memory for assertions."""

    events: list[dict[str, object]]

    def publish(self, event: dict[str, object]) -> None:
        """Record one event payload for test verification."""
        self.events.append(event)


class RaisingExtractor:
    """Raise runtime errors to exercise failed service outcomes."""

    def extract(self, *, chunk: DocumentChunk) -> list[EntityMention]:
        """Raise one extraction failure for test behavior coverage."""
        _ = chunk
        raise RuntimeError("extract failed")

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Return zero sentence count for failed extraction path."""
        _ = chunk
        return 0


def _sample_chunk() -> DocumentChunk:
    """Build one representative chunk fixture for service tests."""
    return DocumentChunk(
        chunk_id="chunk_123",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        chunk_index=0,
        start_char=0,
        end_char=42,
        chunk_text="The Climate Act is discussed by Australia Institute.",
    )


def _zero_entity_type_counts() -> dict[str, int]:
    """Build zeroed per-type counts for compact status payload checks."""
    return {
        "POLICY": 0,
        "ORG": 0,
        "PERSON": 0,
        "JURISDICTION": 0,
        "PROGRAM": 0,
    }


def test_entities_service_processes_one_chunk_and_publishes_status() -> None:
    """Load chunk, persist entities, and publish one completed status event."""
    chunk = _sample_chunk()
    publisher = RecordingPublisher(events=[])
    entity_repo = InMemoryEntityRepository(persisted=[])
    service = EntitiesService(
        chunk_repository=InMemoryChunkRepository(chunks={chunk.chunk_id: chunk}),
        entity_repository=entity_repo,
        extractor=FixedExtractor(),
        event_publisher=publisher,
    )

    report = service.extract_entities_for_chunk(chunk_id=chunk.chunk_id)

    assert report.chunk_id == chunk.chunk_id
    assert report.status == "completed"
    assert report.entities_extracted == 2
    assert report.processed_sentences == 2
    assert report.entities_by_type == {
        "POLICY": 1,
        "ORG": 1,
        "PERSON": 0,
        "JURISDICTION": 0,
        "PROGRAM": 0,
    }
    assert report.error_message is None
    assert len(entity_repo.persisted) == 2
    assert len(publisher.events) == 1
    event = publisher.events[0]
    assert set(event) == {
        "topic",
        "chunk_id",
        "status",
        "entities_extracted",
        "processed_sentences",
        "entities_by_type",
    }
    assert event == {
        "topic": "entities.extraction.status",
        "chunk_id": chunk.chunk_id,
        "status": "completed",
        "entities_extracted": 2,
        "processed_sentences": 2,
        "entities_by_type": {
            "POLICY": 1,
            "ORG": 1,
            "PERSON": 0,
            "JURISDICTION": 0,
            "PROGRAM": 0,
        },
    }


def test_entities_service_returns_failed_when_chunk_missing() -> None:
    """Return failed report and publish status when chunk is missing."""
    publisher = RecordingPublisher(events=[])
    service = EntitiesService(
        chunk_repository=InMemoryChunkRepository(chunks={}),
        entity_repository=InMemoryEntityRepository(persisted=[]),
        extractor=FixedExtractor(),
        event_publisher=publisher,
    )

    report = service.extract_entities_for_chunk(chunk_id="unknown")

    assert report.chunk_id == "unknown"
    assert report.status == "failed"
    assert report.entities_extracted == 0
    assert report.processed_sentences == 0
    assert report.entities_by_type == _zero_entity_type_counts()
    assert report.error_message == "chunk not found"
    assert service.entity_repository.persisted == []
    assert len(publisher.events) == 1
    event = publisher.events[0]
    assert set(event) == {
        "topic",
        "chunk_id",
        "status",
        "entities_extracted",
        "processed_sentences",
        "entities_by_type",
    }
    assert event == {
        "topic": "entities.extraction.status",
        "chunk_id": "unknown",
        "status": "failed",
        "entities_extracted": 0,
        "processed_sentences": 0,
        "entities_by_type": _zero_entity_type_counts(),
    }


def test_entities_service_returns_failed_when_extraction_raises() -> None:
    """Return failed report and publish status on extraction exceptions."""
    chunk = _sample_chunk()
    publisher = RecordingPublisher(events=[])
    service = EntitiesService(
        chunk_repository=InMemoryChunkRepository(chunks={chunk.chunk_id: chunk}),
        entity_repository=InMemoryEntityRepository(persisted=[]),
        extractor=RaisingExtractor(),
        event_publisher=publisher,
    )

    report = service.extract_entities_for_chunk(chunk_id=chunk.chunk_id)

    assert report.chunk_id == chunk.chunk_id
    assert report.status == "failed"
    assert report.entities_extracted == 0
    assert report.processed_sentences == 0
    assert report.entities_by_type == _zero_entity_type_counts()
    assert report.error_message == "extract failed"
    assert service.entity_repository.persisted == []
    assert len(publisher.events) == 1
    event = publisher.events[0]
    assert set(event) == {
        "topic",
        "chunk_id",
        "status",
        "entities_extracted",
        "processed_sentences",
        "entities_by_type",
    }
    assert event == {
        "topic": "entities.extraction.status",
        "chunk_id": chunk.chunk_id,
        "status": "failed",
        "entities_extracted": 0,
        "processed_sentences": 0,
        "entities_by_type": _zero_entity_type_counts(),
    }
