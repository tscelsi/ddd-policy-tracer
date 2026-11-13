"""Services for catalog entity management and promotion."""

from ddd_policy_tracer.__DEPRECATED_analysis.entities._catalog.ports import (
    CatalogCandidateRepository,
    CatalogEntityRepository,
)
from ddd_policy_tracer.enrichment.embedding import AbstractEmbeddingService
from ddd_policy_tracer.__DEPRECATED_analysis.entities.models import (
    CatalogCandidate,
    CatalogEntity,
    EntityType,
    ReviewState,
)
from ddd_policy_tracer.utils.events.local import LocalPublisher


async def list_candidates(
    status: ReviewState | None,
    cc_repo: CatalogCandidateRepository,
) -> list[CatalogCandidate]:
    """List catalog candidates with optional filtering by review status."""
    candidates = await cc_repo.list_candidates(status=status)
    return candidates


async def promote(
    entity_id: str,
    key: str,
    name: str,
    entity_type: EntityType,
    description: str | None,
    cc_repo: CatalogCandidateRepository,
    ce_repo: CatalogEntityRepository,
    event_publisher: LocalPublisher,
) -> None:
    """Promote a catalog candidate to a new catalog entity."""
    candidate = await cc_repo.get_candidate_by_entity_id(entity_id=entity_id)
    if candidate is None:
        raise ValueError(f"Candidate with entity_id {entity_id} not found for promotion.")
    new_entity = candidate.promote_by_creation(
        key=key,
        name=name,
        entity_type=entity_type,
        description=description,
    )
    await ce_repo.save(new_entity)
    event_publisher.publish(
        {
            "topic": "catalog.entity.created",
            "canonical_entity_key": new_entity.canonical_entity_key,
            "canonical_name": new_entity.canonical_name,
            "entity_type": new_entity.entity_type,
        },
    )
    await cc_repo.save(candidate)
    event_publisher.publish(
        {
            "topic": "catalog.candidate.promoted",
            "entity_id": candidate.entity_id,
            "canonical_entity_key": new_entity.canonical_entity_key,
        },
    )


async def add_catalog_entity(
    key: str,
    name: str,
    entity_type: EntityType,
    description: str | None,
    ce_repo: CatalogEntityRepository,
    event_publisher: LocalPublisher,
) -> None:
    """Add a new catalog entity without promotion from a candidate."""
    new_entity = CatalogEntity(
        canonical_entity_key=key,
        canonical_name=name,
        entity_type=entity_type,
        description=description,
    )
    await ce_repo.save(new_entity)
    event_publisher.publish(
        {
            "topic": "catalog.entity.created",
            "canonical_entity_key": new_entity.canonical_entity_key,
            "canonical_name": new_entity.canonical_name,
            "entity_type": new_entity.entity_type,
        },
    )


async def reject_candidate(
    entity_id: str,
    cc_repo: CatalogCandidateRepository,
    event_publisher: LocalPublisher,
) -> None:
    """Reject a catalog candidate by updating its review state."""
    candidate = await cc_repo.get_candidate_by_entity_id(entity_id=entity_id)
    if candidate is None:
        raise ValueError(f"Candidate with entity_id {entity_id} not found for rejection.")
    candidate.set_review_state("rejected")
    await cc_repo.save(candidate)
    event_publisher.publish(
        {
            "topic": "catalog.candidate.rejected",
            "entity_id": candidate.entity_id,
        },
    )


async def reject_candidates(
    entity_ids: list[str],
    cc_repo: CatalogCandidateRepository,
    event_publisher: LocalPublisher,
) -> None:
    """Reject a catalog candidate by updating its review state."""
    candidates = await cc_repo.list_candidates_by_entity_ids(entity_ids=entity_ids)
    for c in candidates:
        c.set_review_state("rejected")
    await cc_repo.save_bulk(candidates)
    event_publisher.publish(
        {
            "topic": "catalog.candidates.rejected",
            "entity_ids": entity_ids,
        },
    )


async def update_existing_catalog_entity(
    canonical_entity_key: str,
    entity_id: str,
    aliases: list[str],
    contexts: list[str],
    description: str | None,
    cc_repo: CatalogCandidateRepository,
    ce_repo: CatalogEntityRepository,
    event_publisher: LocalPublisher,
) -> None:
    """Update an existing catalog entity with new aliases and contexts."""
    candidate = await cc_repo.get_candidate_by_entity_id(entity_id=entity_id)
    if candidate is None:
        raise ValueError(f"Candidate with entity_id {entity_id} not found for update.")
    entity = await ce_repo.get_catalog_entity_by_key(key=canonical_entity_key)
    if entity is None:
        raise ValueError(f"Catalog entity with key {canonical_entity_key} not found for update.")
    candidate.set_review_state("merged")
    entity.add_aliases(aliases)
    entity.add_contexts(contexts)
    entity.update_description(description)
    await ce_repo.save(entity)
    event_publisher.publish(
        {
            "topic": "catalog.entity.updated",
            "canonical_entity_key": entity.canonical_entity_key,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
        },
    )
    await cc_repo.save(candidate)
    event_publisher.publish(
        {
            "topic": "catalog.candidate.merged",
            "entity_id": candidate.entity_id,
        },
    )


async def generate_embedding_for_catalog_entity(
    canonical_entity_key: str,
    ce_repo: CatalogEntityRepository,
    event_publisher: LocalPublisher,
    embedding_service: AbstractEmbeddingService,
) -> None:
    """Generate an embedding for a catalog entity."""
    entity = await ce_repo.get_catalog_entity_by_key(key=canonical_entity_key)
    if entity is None:
        raise ValueError(f"Catalog entity with key {canonical_entity_key} not found for embedding.")
    embedding = embedding_service.embed_one(str(entity))
    entity.update_embedding(embedding)
    await ce_repo.save(entity)
    event_publisher.publish(
        {
            "topic": "catalog.entity.embedding_updated",
            "canonical_entity_key": entity.canonical_entity_key,
        },
    )


async def list_catalog_entities(
    ce_repo: CatalogEntityRepository,
) -> list[CatalogEntity]:
    """List all catalog entities."""
    entities = await ce_repo.list_entities()
    return entities
