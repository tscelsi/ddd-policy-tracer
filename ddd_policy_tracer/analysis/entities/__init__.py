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
    RobustEnsembleEntityExtractor,
    RobustEnsembleEntityExtractorConfig,
    RuleBasedEntityExtractorConfig,
    RuleBasedSentenceEntityExtractor,
    SpacyEntityExtractor,
    SpacyEntityExtractorConfig,
)
from .models import EntityExtractionReport, EntityMention, EntityType
from .ports import (
    ChunkRepository,
    EntityExtractor,
    EntityRepository,
    EventPublisher,
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
    AutoApprovalReport,
    PromotionCandidate,
    ReviewDecisionReport,
    ReviewQueueWriteReport,
    SQLiteReviewQueueRepository,
)
from .service_layer import EntitiesService

__all__ = [
    "AutoApprovalReport",
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
    "EntityRepository",
    "EntityResolutionService",
    "EntityType",
    "EventPublisher",
    "FilesystemChunkRepository",
    "FilesystemEntityRepository",
    "HybridCatalogRetriever",
    "PromotionCandidate",
    "ResolutionDecision",
    "RetrievalCandidate",
    "RetrievalResult",
    "ReviewDecisionReport",
    "ReviewQueueWriteReport",
    "RobustEnsembleEntityExtractor",
    "RobustEnsembleEntityExtractorConfig",
    "RuleBasedEntityExtractorConfig",
    "RuleBasedSentenceEntityExtractor",
    "SQLiteReviewQueueRepository",
    "SpacyEntityExtractor",
    "SpacyEntityExtractorConfig",
    "apply_resolution_decision",
    "get_catalog_metadata",
    "import_seed_catalog",
    "promote_review_items",
]
