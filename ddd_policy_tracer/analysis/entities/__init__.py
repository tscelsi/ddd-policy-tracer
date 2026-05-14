"""Public exports for entities bounded context."""

from .adapters import FilesystemChunkRepository, FilesystemEntityRepository
from .catalog import CatalogImportReport, get_catalog_metadata, import_seed_catalog
from .extractors import (
    RobustEnsembleEntityExtractor,
    RobustEnsembleEntityExtractorConfig,
    RuleBasedEntityExtractorConfig,
    RuleBasedSentenceEntityExtractor,
    SpacyFastCorefEntityExtractor,
    SpacyFastCorefEntityExtractorConfig,
)
from .models import EntityExtractionReport, EntityMention, EntityType
from .resolution import (
    CandidateEvidence,
    DecisionStatus,
    DeterministicEntityJudge,
    EntityResolutionService,
    ResolutionDecision,
    apply_resolution_decision,
)
from .retrieval import HybridCatalogRetriever, RetrievalCandidate, RetrievalResult
from .ports import (
    ChunkRepository,
    EntityExtractor,
    EntityRepository,
    EventPublisher,
)
from .service_layer import EntitiesService

__all__ = [
    "ChunkRepository",
    "CatalogImportReport",
    "EntitiesService",
    "EntityExtractionReport",
    "EntityExtractor",
    "EntityMention",
    "EntityRepository",
    "EntityType",
    "EventPublisher",
    "FilesystemChunkRepository",
    "FilesystemEntityRepository",
    "get_catalog_metadata",
    "import_seed_catalog",
    "RobustEnsembleEntityExtractor",
    "RobustEnsembleEntityExtractorConfig",
    "RuleBasedEntityExtractorConfig",
    "RuleBasedSentenceEntityExtractor",
    "HybridCatalogRetriever",
    "RetrievalCandidate",
    "RetrievalResult",
    "CandidateEvidence",
    "DecisionStatus",
    "DeterministicEntityJudge",
    "EntityResolutionService",
    "ResolutionDecision",
    "apply_resolution_decision",
    "SpacyFastCorefEntityExtractor",
    "SpacyFastCorefEntityExtractorConfig",
]
