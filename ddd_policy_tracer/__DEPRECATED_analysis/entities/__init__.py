"""Public exports for entities bounded context."""

from .adapters import FilesystemChunkRepository, FilesystemEntityRepository
from .catalog import (
    CatalogImportReport,
    CatalogPromotionReport,
    get_catalog_metadata,
    import_seed_catalog,
    promote_review_items,
)
from .extractors import (
    SpacyEntityExtractor,
    SpacyEntityExtractorConfig,
)
from .models import EntityExtractionReport, EntityMention, EntityType
from .ports import (
    ChunkRepository,
    EntityExtractor,
    EntityMentionRepository,
)
from .resolution import (
    CandidateEvidence,
    DecisionStatus,
    DeterministicEntityJudge,
    EntityResolutionService,
    ResolutionDecision,
    apply_resolution_decision,
)
from .retrieval import HybridCatalogRetriever, RetrievalCandidate, RetrievalResult
from .review_queue import (
    PromotionCandidate,
    ReviewDecisionReport,
    ReviewQueueWriteReport,
    SQLiteReviewQueueRepository,
)
from .service_layer import EntitiesService

__all__ = [
    "CandidateEvidence",
    "CatalogImportReport",
    "CatalogPromotionReport",
    "ChunkRepository",
    "DecisionStatus",
    "DeterministicEntityJudge",
    "EntitiesService",
    "EntityExtractionReport",
    "EntityExtractor",
    "EntityMention",
    "EntityMentionRepository",
    "EntityResolutionService",
    "EntityType",
    "FilesystemChunkRepository",
    "FilesystemEntityRepository",
    "HybridCatalogRetriever",
    "PromotionCandidate",
    "ResolutionDecision",
    "RetrievalCandidate",
    "RetrievalResult",
    "ReviewDecisionReport",
    "ReviewQueueWriteReport",
    "SQLiteReviewQueueRepository",
    "SpacyEntityExtractor",
    "SpacyEntityExtractorConfig",
    "apply_resolution_decision",
    "get_catalog_metadata",
    "import_seed_catalog",
    "promote_review_items",
]
