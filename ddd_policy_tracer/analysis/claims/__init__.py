"""Public exports for claims bounded context."""

from .adapters import FilesystemClaimRepository
from .models import ClaimCandidate, ClaimExtractionReport
from .ports import (
    ChunkRepository,
    ClaimExtractor,
    ClaimRepository,
    EventPublisher,
)
from .service_layer import ClaimsService

__all__ = [
    "ChunkRepository",
    "ClaimCandidate",
    "ClaimExtractionReport",
    "ClaimExtractor",
    "ClaimRepository",
    "ClaimsService",
    "EventPublisher",
    "FilesystemClaimRepository",
]
