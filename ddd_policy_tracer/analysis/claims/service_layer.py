"""Application service orchestration for claims bounded context."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ClaimExtractionReport
from .ports import (
    ChunkRepository,
    ClaimExtractor,
    ClaimRepository,
    EventPublisher,
)


@dataclass(frozen=True)
class ClaimsService:
    """Coordinate chunk loading, extraction, persistence, and eventing."""

    chunk_repository: ChunkRepository
    claim_repository: ClaimRepository
    extractor: ClaimExtractor
    event_publisher: EventPublisher

    def extract_claims_for_chunk(self, *, chunk_id: str) -> ClaimExtractionReport:
        """Extract and persist claims for one chunk identity."""
        chunk = self.chunk_repository.get_chunk(chunk_id=chunk_id)
        if chunk is None:
            return self._publish_failure(
                chunk_id=chunk_id,
                processed_sentences=0,
                error_message="chunk not found",
            )

        try:
            processed_sentences = self.extractor.count_processed_sentences(
                chunk=chunk,
            )
            claims = self.extractor.extract(chunk=chunk)
            claims_extracted = self.claim_repository.add_claims(claims)
        except Exception as exc:
            return self._publish_failure(
                chunk_id=chunk_id,
                processed_sentences=0,
                error_message=str(exc),
            )

        report = ClaimExtractionReport(
            chunk_id=chunk_id,
            status="completed",
            claims_extracted=claims_extracted,
            processed_sentences=processed_sentences,
            error_message=None,
        )
        self.event_publisher.publish(
            {
                "topic": "claims.extraction.status",
                "chunk_id": chunk_id,
                "status": report.status,
                "claims_extracted": report.claims_extracted,
                "processed_sentences": report.processed_sentences,
            },
        )
        return report

    def _publish_failure(
        self,
        *,
        chunk_id: str,
        processed_sentences: int,
        error_message: str,
    ) -> ClaimExtractionReport:
        """Publish a failed status event and return a failure report."""
        report = ClaimExtractionReport(
            chunk_id=chunk_id,
            status="failed",
            claims_extracted=0,
            processed_sentences=processed_sentences,
            error_message=error_message,
        )
        self.event_publisher.publish(
            {
                "topic": "claims.extraction.status",
                "chunk_id": chunk_id,
                "status": report.status,
                "claims_extracted": report.claims_extracted,
                "processed_sentences": report.processed_sentences,
            },
        )
        return report
