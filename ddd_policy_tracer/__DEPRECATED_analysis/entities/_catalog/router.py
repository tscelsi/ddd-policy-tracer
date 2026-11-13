"""FastAPI router for catalog entity management and promotion services."""

import logging
from typing import Annotated, cast

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel

from ddd_policy_tracer.__DEPRECATED_analysis.entities._catalog.adapters import (
    SQLiteCatalogCandidateRepository,
    SQLiteCatalogEntityRepository,
)
from ddd_policy_tracer.__DEPRECATED_analysis.entities._catalog.service_layer import (
    add_catalog_entity,
    generate_embedding_for_catalog_entity,
    promote,
    reject_candidates,
    update_existing_catalog_entity,
)
from ddd_policy_tracer.__DEPRECATED_analysis.entities._catalog.service_layer import (
    list_catalog_entities as _list_catalog_entities,
)
from ddd_policy_tracer.enrichment.embedding import AbstractEmbeddingService
from ddd_policy_tracer.__DEPRECATED_analysis.entities.models import (
    CatalogCandidate,
    CatalogEntity,
    EntityType,
    ReviewState,
)
from ddd_policy_tracer.utils.events.local import LocalPublisher

logger = logging.getLogger(__name__)
# --- Pydantic Schemas ---


class PromoteRequest(BaseModel):
    """Request schema for promoting a catalog candidate to a new catalog entity."""

    entity_id: str
    key: str
    name: str
    entity_type: EntityType
    description: str | None = None


class RejectBulkRequest(BaseModel):
    """Request schema for rejecting catalog candidates by their entity identities."""

    entity_ids: list[str]


class UpdateEntityRequest(BaseModel):
    """Request schema for updating an existing catalog entity with new aliases and contexts."""

    canonical_entity_key: str
    entity_id: str
    aliases: list[str]
    contexts: list[str]


# --- Repository Instances (for demo, use a fixed path) ---
publisher = LocalPublisher()


def ce_repo_dep() -> SQLiteCatalogEntityRepository:
    """Dependency function to provide the catalog entity repository instance."""
    return SQLiteCatalogEntityRepository()


def cc_repo_dep() -> SQLiteCatalogCandidateRepository:
    """Dependency function to provide the catalog candidate repository instance."""
    return SQLiteCatalogCandidateRepository()


# --- FastAPI Router ---

router = APIRouter(prefix="/catalog", tags=["catalog"])
entity_router = APIRouter(prefix="/entities")
candidate_router = APIRouter(prefix="/candidates")


@candidate_router.get("", response_model=list[CatalogCandidate])
async def list_catalog_candidates(
    cc_repo: Annotated[SQLiteCatalogCandidateRepository, Depends(cc_repo_dep)],
    status: ReviewState | None = None,
) -> list[CatalogCandidate]:
    """List catalog candidates with optional filtering by review status."""
    candidates = await cc_repo.list_candidates(status=status)
    return candidates


@candidate_router.post("/promote")
async def promote_candidate(
    req: PromoteRequest,
    ce_repo: Annotated[SQLiteCatalogEntityRepository, Depends(ce_repo_dep)],
    cc_repo: Annotated[SQLiteCatalogCandidateRepository, Depends(cc_repo_dep)],
) -> dict[str, str]:
    """Promote a catalog candidate to a new catalog entity."""
    try:
        await promote(
            entity_id=req.entity_id,
            key=req.key,
            name=req.name,
            entity_type=req.entity_type,
            description=req.description,
            cc_repo=cc_repo,
            ce_repo=ce_repo,
            event_publisher=publisher,
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@candidate_router.post("/reject")
async def reject_catalog_candidates(
    req: RejectBulkRequest,
    cc_repo: Annotated[SQLiteCatalogCandidateRepository, Depends(cc_repo_dep)],
) -> dict[str, list[str]]:
    """Reject multiple catalog candidates by their entity mention identities."""
    try:
        await reject_candidates(
            entity_ids=req.entity_ids,
            cc_repo=cc_repo,
            event_publisher=publisher,
        )
        return {"status": "success", "rejected": req.entity_ids}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@entity_router.post("")
async def create_catalog_entity(
    ce_repo: Annotated[SQLiteCatalogEntityRepository, Depends(ce_repo_dep)],
    key: str = Body(..., embed=True),
    name: str = Body(..., embed=True),
    entity_type: EntityType = Body(..., embed=True),  # noqa: B008
    description: str | None = Body(None, embed=True),
) -> dict[str, str]:
    """List catalog candidates with optional filtering by review status."""
    await add_catalog_entity(
        key=key,
        name=name,
        entity_type=entity_type,
        description=description,
        ce_repo=ce_repo,
        event_publisher=publisher,
    )
    return {"status": "success"}


@entity_router.get("/")
async def list_catalog_entities(
    ce_repo: Annotated[SQLiteCatalogEntityRepository, Depends(ce_repo_dep)],
) -> list[CatalogEntity]:
    """List all catalog entities."""
    entities = await _list_catalog_entities(ce_repo=ce_repo)
    return entities


@entity_router.put("")
async def update_catalog_entity(
    req: UpdateEntityRequest,
    ce_repo: Annotated[SQLiteCatalogEntityRepository, Depends(ce_repo_dep)],
    cc_repo: Annotated[SQLiteCatalogCandidateRepository, Depends(cc_repo_dep)],
) -> dict[str, str]:
    """Update an existing catalog entity with new aliases and contexts."""
    try:
        await update_existing_catalog_entity(
            canonical_entity_key=req.canonical_entity_key,
            entity_id=req.entity_id,
            aliases=req.aliases,
            contexts=req.contexts,
            cc_repo=cc_repo,
            ce_repo=ce_repo,
            event_publisher=publisher,
        )
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@entity_router.put("/embed")
async def embed_catalog_entity(
    canonical_entity_key: str,
    req: Request,
    ce_repo: Annotated[SQLiteCatalogEntityRepository, Depends(ce_repo_dep)],
) -> dict[str, str]:
    """Update an existing catalog entity with new aliases and contexts."""
    try:
        embed_service = cast(AbstractEmbeddingService, req.state.embed_service)
        await generate_embedding_for_catalog_entity(
            canonical_entity_key=canonical_entity_key,
            ce_repo=ce_repo,
            event_publisher=publisher,
            embedding_service=embed_service,
        )
        return {"status": "success"}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=400, detail=str(e)) from e


router.include_router(entity_router)
router.include_router(candidate_router)
