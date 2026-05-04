"""Unit tests for claims service tracer-bullet orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from ddd_policy_tracer.analysis.chunking_models import DocumentChunk
from ddd_policy_tracer.analysis.claims import ClaimCandidate, ClaimsService


@dataclass
class InMemoryChunkRepository:
    """Provide predictable chunk lookup by chunk identifier."""

    chunks: dict[str, DocumentChunk]

    def get_chunk(self, *, chunk_id: str) -> DocumentChunk | None:
        """Return one chunk object when present in memory."""
        return self.chunks.get(chunk_id)


@dataclass
class InMemoryClaimRepository:
    """Collect persisted claims in memory for service assertions."""

    persisted: list[ClaimCandidate]

    def add_claims(self, claims: list[ClaimCandidate]) -> int:
        """Persist claim candidates and report inserted count."""
        self.persisted.extend(claims)
        return len(claims)


class FixedExtractor:
    """Return deterministic claim and sentence counts for one chunk."""

    def extract(self, *, chunk: DocumentChunk) -> list[ClaimCandidate]:
        """Return one fixed claim candidate for any chunk."""
        return [ClaimCandidate(evidence_text=chunk.chunk_text)]

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Return deterministic sentence count for report output."""
        _ = chunk
        return 1


@dataclass
class RecordingPublisher:
    """Record published events in memory for assertions."""

    events: list[dict[str, object]]

    def publish(self, event: dict[str, object]) -> None:
        """Record one event payload for test verification."""
        self.events.append(event)


class RaisingExtractor:
    """Raise runtime errors to exercise failed service outcomes."""

    def extract(self, *, chunk: DocumentChunk) -> list[ClaimCandidate]:
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
        chunk_text="Policy settings should reduce emissions by 4.9%.",
    )


def test_claims_service_processes_one_chunk_and_publishes_status() -> None:
    """Load chunk, persist claims, and publish one completed status event."""
    chunk = _sample_chunk()
    publisher = RecordingPublisher(events=[])
    claim_repo = InMemoryClaimRepository(persisted=[])
    service = ClaimsService(
        chunk_repository=InMemoryChunkRepository(chunks={chunk.chunk_id: chunk}),
        claim_repository=claim_repo,
        extractor=FixedExtractor(),
        event_publisher=publisher,
    )

    report = service.extract_claims_for_chunk(chunk_id=chunk.chunk_id)

    assert report.chunk_id == chunk.chunk_id
    assert report.status == "completed"
    assert report.claims_extracted == 1
    assert report.processed_sentences == 1
    assert report.error_message is None
    assert len(claim_repo.persisted) == 1
    assert publisher.events == [
        {
            "topic": "claims.extraction.status",
            "chunk_id": chunk.chunk_id,
            "status": "completed",
            "claims_extracted": 1,
            "processed_sentences": 1,
        },
    ]


def test_claims_service_returns_failed_when_chunk_missing() -> None:
    """Return failed report and publish status when chunk is missing."""
    publisher = RecordingPublisher(events=[])
    service = ClaimsService(
        chunk_repository=InMemoryChunkRepository(chunks={}),
        claim_repository=InMemoryClaimRepository(persisted=[]),
        extractor=FixedExtractor(),
        event_publisher=publisher,
    )

    report = service.extract_claims_for_chunk(chunk_id="unknown")

    assert report.chunk_id == "unknown"
    assert report.status == "failed"
    assert report.claims_extracted == 0
    assert report.processed_sentences == 0
    assert report.error_message == "chunk not found"
    assert publisher.events == [
        {
            "topic": "claims.extraction.status",
            "chunk_id": "unknown",
            "status": "failed",
            "claims_extracted": 0,
            "processed_sentences": 0,
        },
    ]


def test_claims_service_returns_failed_when_extraction_raises() -> None:
    """Return failed report and publish status on extraction exceptions."""
    chunk = _sample_chunk()
    publisher = RecordingPublisher(events=[])
    service = ClaimsService(
        chunk_repository=InMemoryChunkRepository(chunks={chunk.chunk_id: chunk}),
        claim_repository=InMemoryClaimRepository(persisted=[]),
        extractor=RaisingExtractor(),
        event_publisher=publisher,
    )

    report = service.extract_claims_for_chunk(chunk_id=chunk.chunk_id)

    assert report.chunk_id == chunk.chunk_id
    assert report.status == "failed"
    assert report.claims_extracted == 0
    assert report.processed_sentences == 0
    assert report.error_message == "extract failed"
    assert publisher.events == [
        {
            "topic": "claims.extraction.status",
            "chunk_id": chunk.chunk_id,
            "status": "failed",
            "claims_extracted": 0,
            "processed_sentences": 0,
        },
    ]
