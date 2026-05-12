"""Public exports for claims bounded context."""

from .adapters import FilesystemChunkRepository, FilesystemClaimRepository
from .extractors import (
    HuggingFaceClaimExtractor,
    HuggingFaceClaimExtractorConfig,
    LLMClaimExtractor,
    LLMClaimExtractorConfig,
    MLClaimExtractor,
    MLClaimExtractorConfig,
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
    "HuggingFaceClaimExtractor",
    "HuggingFaceClaimExtractorConfig",
    "LLMClaimExtractor",
    "LLMClaimExtractorConfig",
    "MLClaimExtractor",
    "MLClaimExtractorConfig",
    "RuleBasedClaimExtractorConfig",
    "RuleBasedSentenceClaimExtractor",
]
