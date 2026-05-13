"""Application services for analysis canonicalization stage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Protocol

from ddd_policy_tracer.analysis.claims.models import ClaimCandidate
from ddd_policy_tracer.analysis.entities.models import EntityMention

from .adapters import rows_to_staleness_fingerprint
from .domain import canonicalize_claims, canonicalize_entities
from .models import CanonicalClaimRecord, CanonicalEntityRecord, CanonicalizationReport


class EntitySourceRepository(Protocol):
    """Port contract for loading source entities and input hash."""

    def list_entities(self) -> list[EntityMention]:
        """Load source entity rows for canonicalization."""
        raise NotImplementedError

    def artifact_hash(self) -> str:
        """Return deterministic source artifact hash value."""
        raise NotImplementedError


class ClaimSourceRepository(Protocol):
    """Port contract for loading source claims and input hash."""

    def list_claims(self) -> list[ClaimCandidate]:
        """Load source claim rows for canonicalization."""
        raise NotImplementedError

    def artifact_hash(self) -> str:
        """Return deterministic source artifact hash value."""
        raise NotImplementedError


class CanonicalEntityRepository(Protocol):
    """Port contract for persisting canonical entity rows."""

    def replace_all(self, *, rows: list[CanonicalEntityRecord]) -> None:
        """Replace canonical entity output with provided rows."""
        raise NotImplementedError

    def list_entities(self) -> list[CanonicalEntityRecord]:
        """List canonical entity rows for claim-stage dependency."""
        raise NotImplementedError

    def artifact_hash(self) -> str:
        """Return deterministic artifact hash for canonical entities."""
        raise NotImplementedError


class CanonicalClaimRepository(Protocol):
    """Port contract for persisting canonical claim rows."""

    def replace_all(self, *, rows: list[CanonicalClaimRecord]) -> None:
        """Replace canonical claim output with provided rows."""
        raise NotImplementedError


@dataclass(frozen=True)
class EntityCanonicalizationService:
    """Canonicalize source entities and persist canonical entity artifact."""

    source_repository: EntitySourceRepository
    canonical_repository: CanonicalEntityRepository
    canonicalizer_version: str

    def run(self) -> CanonicalizationReport:
        """Execute one full entity canonicalization pass."""
        entities = self.source_repository.list_entities()
        input_hash = self.source_repository.artifact_hash()
        generated_at = _deterministic_generated_at(
            stage="entity",
            canonicalizer_version=self.canonicalizer_version,
            input_artifact_hash=input_hash,
        )
        result = canonicalize_entities(
            entities=entities,
            canonicalizer_version=self.canonicalizer_version,
            generated_at=generated_at,
            input_artifact_hash=input_hash,
        )
        self.canonical_repository.replace_all(rows=result.rows)
        return CanonicalizationReport(
            stage="entity",
            input_rows=len(entities),
            output_rows=len(result.rows),
            canonicalizer_version=self.canonicalizer_version,
            generated_at=generated_at,
        )


@dataclass(frozen=True)
class ClaimCanonicalizationService:
    """Canonicalize source claims by linking canonical entities."""

    claim_source_repository: ClaimSourceRepository
    canonical_entity_repository: CanonicalEntityRepository
    canonical_claim_repository: CanonicalClaimRepository
    claim_canonicalizer_version: str
    required_entity_canonicalizer_version: str

    def run(self) -> CanonicalizationReport:
        """Execute one full claim canonicalization pass with fail-fast checks."""
        claims = self.claim_source_repository.list_claims()
        canonical_entities = self.canonical_entity_repository.list_entities()
        if not canonical_entities:
            raise ValueError("canonical entities input is missing")

        source_ids, checksums, version = rows_to_staleness_fingerprint(rows=canonical_entities)
        claim_source_ids = {claim.source_id for claim in claims}
        claim_checksums = {claim.document_checksum for claim in claims}
        if source_ids != claim_source_ids:
            raise ValueError("canonical entities are stale: source_id coverage mismatch")
        if checksums != claim_checksums:
            raise ValueError("canonical entities are stale: document_checksum coverage mismatch")
        if version != self.required_entity_canonicalizer_version:
            raise ValueError("canonical entities are stale: canonicalizer version mismatch")

        input_hash = "|".join(
            [
                self.claim_source_repository.artifact_hash(),
                self.canonical_entity_repository.artifact_hash(),
            ],
        )
        generated_at = _deterministic_generated_at(
            stage="claim",
            canonicalizer_version=self.claim_canonicalizer_version,
            input_artifact_hash=input_hash,
        )
        result = canonicalize_claims(
            claims=claims,
            canonical_entities=canonical_entities,
            canonicalizer_version=self.claim_canonicalizer_version,
            generated_at=generated_at,
            input_artifact_hash=input_hash,
        )
        self.canonical_claim_repository.replace_all(rows=result.rows)
        return CanonicalizationReport(
            stage="claim",
            input_rows=len(claims),
            output_rows=len(result.rows),
            canonicalizer_version=self.claim_canonicalizer_version,
            generated_at=generated_at,
        )


def _deterministic_generated_at(
    *,
    stage: str,
    canonicalizer_version: str,
    input_artifact_hash: str,
) -> str:
    """Build deterministic pseudo-timestamp from versioned stage inputs."""
    raw = f"{stage}|{canonicalizer_version}|{input_artifact_hash}"
    digest = sha256(raw.encode("utf-8")).hexdigest()
    seconds = int(digest[:12], 16) % 2_000_000_000
    return datetime.fromtimestamp(seconds, tz=UTC).isoformat()
