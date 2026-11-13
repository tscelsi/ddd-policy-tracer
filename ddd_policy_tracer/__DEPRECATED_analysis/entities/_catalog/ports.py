"""Ports for entity extraction, resolution, and review."""

from typing import Protocol

from ddd_policy_tracer.__DEPRECATED_analysis.entities.models import CatalogCandidate, CatalogEntity
from ddd_policy_tracer.__DEPRECATED_analysis.entities.review_queue import ReviewState


class CatalogEntityRepository(Protocol):
    """Load and persist catalog entities for entity resolution and linking."""

    async def list_entities(self) -> list[CatalogEntity]:
        """Load all catalog entities as entity mentions for resolution."""

    async def get_catalog_entity_by_key(self, *, key: str) -> CatalogEntity | None:
        """Return one catalog entity by canonical key or none when missing."""

    async def save(self, entity: CatalogEntity) -> None:
        """Promote."""


class CatalogCandidateRepository(Protocol):
    """Load and persist catalog candidates for entity resolution and linking."""

    async def list_candidates(self, status: ReviewState | None = None) -> list[CatalogCandidate]:
        """Load catalog candidates for resolution and review."""

    async def get_candidate_by_entity_id(self, *, entity_id: str) -> CatalogCandidate | None:
        """Return one candidate by entity mention identity or none when missing."""

    async def list_candidates_by_entity_ids(
        self, *, entity_ids: list[str]
    ) -> list[CatalogCandidate]:
        """Return candidates linked to any of the provided entity mention identities."""

    async def save(self, candidate: CatalogCandidate) -> None:
        """Add or update a catalog candidate."""

    async def save_bulk(self, candidates: list[CatalogCandidate]) -> None:
        """Add or update many catalog candidates in one batch operation."""
