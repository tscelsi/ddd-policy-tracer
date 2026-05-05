"""Ports for claims bounded context orchestration dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk

from .models import ClaimCandidate


@dataclass(frozen=True)
class ExtractionMetrics:
    """Represent extractor metrics used by service reporting and events."""

    processed_sentences: int


class ChunkRepository(Protocol):
    """Load chunks by identity for claim extraction workflows."""

    def get_chunk(self, *, chunk_id: str) -> DocumentChunk | None:
        """Return one chunk for extraction or none when missing."""


class ClaimRepository(Protocol):
    """Persist extracted claim candidates."""

    def add_claims(self, claims: list[ClaimCandidate]) -> int:
        """Persist claim candidates and return inserted claim count."""

    def list_claims(self, *, chunk_id: str | None = None) -> list[ClaimCandidate]:
        """Load persisted claim candidates, optionally by chunk identity."""


class ClaimExtractor(Protocol):
    """Extract claim candidates from one source chunk."""

    def extract(self, *, chunk: DocumentChunk) -> list[ClaimCandidate]:
        """Return extracted claim candidates for one chunk."""

    def count_processed_sentences(self, *, chunk: DocumentChunk) -> int:
        """Return number of processed sentences for one chunk."""


class EventPublisher(Protocol):
    """Publish compact orchestration status events."""

    def publish(self, event: dict[str, object]) -> None:
        """Publish one event for downstream processing."""
