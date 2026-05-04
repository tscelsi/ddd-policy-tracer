"""Public exports for claims bounded context."""

from .adapters import FilesystemChunkRepository, FilesystemClaimRepository
from .extractors import (
    LLMClaimExtractor,
    LLMClaimExtractorConfig,
    RuleBasedClaimExtractorConfig,
    RuleBasedSentenceClaimExtractor,
)
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
    "FilesystemChunkRepository",
    "FilesystemClaimRepository",
    "LLMClaimExtractor",
    "LLMClaimExtractorConfig",
    "RuleBasedClaimExtractorConfig",
    "RuleBasedSentenceClaimExtractor",
]
