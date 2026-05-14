"""Pure canonicalization domain logic for entities and claims."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256

from ddd_policy_tracer.analysis.claims.models import ClaimCandidate
from ddd_policy_tracer.analysis.entities.models import EntityMention

from .models import (
    CanonicalClaimRecord,
    CanonicalEntityRecord,
    CanonicalizationMetadata,
    LinkedEntityEvidence,
    LinkedEntityRecord,
    PendingEntityLinkRecord,
)


@dataclass(frozen=True)
class EntityCanonicalizationResult:
    """Hold canonicalized entities and coverage metadata."""

    rows: list[CanonicalEntityRecord]
    source_ids: set[str]
    document_checksums: set[str]


@dataclass(frozen=True)
class ClaimCanonicalizationResult:
    """Hold canonicalized claims and coverage metadata."""

    rows: list[CanonicalClaimRecord]
    source_ids: set[str]
    document_checksums: set[str]


def canonicalize_entities(
    *,
    entities: list[EntityMention],
    canonicalizer_version: str,
    generated_at: str,
    input_artifact_hash: str,
) -> EntityCanonicalizationResult:
    """Canonicalize extracted entities with deterministic keys and names."""
    metadata = CanonicalizationMetadata(
        stage="entity",
        canonicalizer_version=canonicalizer_version,
        generated_at=generated_at,
        input_artifact_hash=input_artifact_hash,
    )
    rows: list[CanonicalEntityRecord] = []
    source_ids: set[str] = set()
    document_checksums: set[str] = set()

    for entity in entities:
        canonical_name = _canonicalize_entity_name(entity.normalized_mention_text)
        canonical_entity_key = _build_canonical_entity_key(
            entity_type=entity.entity_type,
            canonical_name=canonical_name,
        )
        rows.append(
            CanonicalEntityRecord(
                entity_id=entity.entity_id,
                chunk_id=entity.chunk_id,
                source_id=entity.source_id,
                source_document_id=entity.source_document_id,
                document_checksum=entity.document_checksum,
                start_char=entity.start_char,
                end_char=entity.end_char,
                mention_text=entity.mention_text,
                normalized_mention_text=entity.normalized_mention_text,
                entity_type=entity.entity_type,
                confidence=entity.confidence,
                extractor_version=entity.extractor_version,
                canonical_name=canonical_name,
                canonical_entity_key=canonical_entity_key,
                decision_status=_decision_status_from_metadata(entity),
                decision_score=_decision_score_from_metadata(entity),
                selected_candidate_key=_selected_candidate_key_from_metadata(entity),
                canonicalization=metadata,
            ),
        )
        source_ids.add(entity.source_id)
        document_checksums.add(entity.document_checksum)

    return EntityCanonicalizationResult(
        rows=rows,
        source_ids=source_ids,
        document_checksums=document_checksums,
    )


def canonicalize_claims(
    *,
    claims: list[ClaimCandidate],
    canonical_entities: list[CanonicalEntityRecord],
    canonicalizer_version: str,
    generated_at: str,
    input_artifact_hash: str,
) -> ClaimCanonicalizationResult:
    """Canonicalize claims by attaching canonical linked entities."""
    metadata = CanonicalizationMetadata(
        stage="claim",
        canonicalizer_version=canonicalizer_version,
        generated_at=generated_at,
        input_artifact_hash=input_artifact_hash,
    )
    entities_by_chunk: dict[str, list[CanonicalEntityRecord]] = {}
    for entity in canonical_entities:
        entities_by_chunk.setdefault(entity.chunk_id, []).append(entity)

    rows: list[CanonicalClaimRecord] = []
    source_ids: set[str] = set()
    document_checksums: set[str] = set()
    for claim in claims:
        chunk_entities = entities_by_chunk.get(claim.chunk_id, [])
        linked_entities = _link_entities_for_claim(claim=claim, entities=chunk_entities)
        immediate_links = [
            linked
            for linked in linked_entities
            if _decision_status_for_entity(linked.entity_id, chunk_entities) == "linked"
        ]
        pending_links = [
            PendingEntityLinkRecord(
                canonical_entity_key=linked.canonical_entity_key,
                entity_type=linked.entity_type,
                canonical_name=linked.canonical_name,
                decision_status=_decision_status_for_entity(linked.entity_id, chunk_entities),
                decision_score=_decision_score_for_entity(linked.entity_id, chunk_entities),
                entity_id=linked.entity_id,
                evidence=linked.evidence,
            )
            for linked in linked_entities
            if _decision_status_for_entity(linked.entity_id, chunk_entities)
            in {"needs_review", "new_candidate", "abstain"}
        ]
        rows.append(
            CanonicalClaimRecord(
                claim_id=claim.claim_id,
                chunk_id=claim.chunk_id,
                source_id=claim.source_id,
                source_document_id=claim.source_document_id,
                document_checksum=claim.document_checksum,
                start_char=claim.start_char,
                end_char=claim.end_char,
                evidence_text=claim.evidence_text,
                normalized_claim_text=claim.normalized_claim_text,
                confidence=claim.confidence,
                claim_type=claim.claim_type,
                extractor_version=claim.extractor_version,
                linked_entities=immediate_links,
                pending_entity_links=pending_links,
                canonicalization=metadata,
            ),
        )
        source_ids.add(claim.source_id)
        document_checksums.add(claim.document_checksum)

    return ClaimCanonicalizationResult(
        rows=rows,
        source_ids=source_ids,
        document_checksums=document_checksums,
    )


def _canonicalize_entity_name(value: str) -> str:
    """Normalize one entity name for deterministic canonical grouping."""
    lowered = value.casefold().strip()
    no_punctuation = re.sub(r"[^a-z0-9\s]", " ", lowered)
    collapsed = " ".join(no_punctuation.split())
    if collapsed.startswith("the "):
        return collapsed[4:]
    return collapsed


def _build_canonical_entity_key(*, entity_type: str, canonical_name: str) -> str:
    """Build deterministic canonical entity key from stable inputs."""
    raw = f"{entity_type}|{canonical_name}"
    digest = sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"entity_canon_{digest}"


def _link_entities_for_claim(
    *,
    claim: ClaimCandidate,
    entities: list[CanonicalEntityRecord],
) -> list[LinkedEntityRecord]:
    """Link canonical entities to one claim by span first, then text fallback."""
    linked: list[LinkedEntityRecord] = []
    linked_entity_ids: set[str] = set()

    for entity in entities:
        if _spans_overlap(
            claim_start=claim.start_char,
            claim_end=claim.end_char,
            entity_start=entity.start_char,
            entity_end=entity.end_char,
        ):
            linked.append(
                LinkedEntityRecord(
                    canonical_entity_key=entity.canonical_entity_key,
                    entity_type=entity.entity_type,
                    canonical_name=entity.canonical_name,
                    link_method="span_overlap",
                    entity_id=entity.entity_id,
                    evidence=LinkedEntityEvidence(
                        claim_span=(claim.start_char, claim.end_char),
                        entity_span=(entity.start_char, entity.end_char),
                        matched_text=None,
                    ),
                ),
            )
            linked_entity_ids.add(entity.entity_id)

    normalized_claim = _normalize_match_value(claim.normalized_claim_text)
    for entity in entities:
        if entity.entity_id in linked_entity_ids:
            continue
        normalized_entity = _normalize_match_value(entity.normalized_mention_text)
        if not normalized_entity or normalized_entity not in normalized_claim:
            continue
        linked.append(
            LinkedEntityRecord(
                canonical_entity_key=entity.canonical_entity_key,
                entity_type=entity.entity_type,
                canonical_name=entity.canonical_name,
                link_method="text_match_fallback",
                entity_id=entity.entity_id,
                evidence=LinkedEntityEvidence(
                    claim_span=(claim.start_char, claim.end_char),
                    entity_span=(entity.start_char, entity.end_char),
                    matched_text=entity.normalized_mention_text,
                ),
            ),
        )
        linked_entity_ids.add(entity.entity_id)

    return linked


def _spans_overlap(*, claim_start: int, claim_end: int, entity_start: int, entity_end: int) -> bool:
    """Return true when claim and entity spans overlap in one chunk."""
    return max(claim_start, entity_start) < min(claim_end, entity_end)


def _normalize_match_value(value: str) -> str:
    """Normalize text for robust containment comparisons."""
    lowered = value.casefold()
    no_punctuation = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(no_punctuation.split())


def _decision_status_from_metadata(entity: EntityMention) -> str:
    """Read resolver decision status from mention metadata with default link."""
    metadata = entity.metadata or {}
    value = metadata.get("decision_status")
    return str(value) if isinstance(value, str) else "linked"


def _decision_score_from_metadata(entity: EntityMention) -> float | None:
    """Read resolver decision score from mention metadata when present."""
    metadata = entity.metadata or {}
    value = metadata.get("decision_score")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _selected_candidate_key_from_metadata(entity: EntityMention) -> str | None:
    """Read selected candidate key from mention metadata when present."""
    metadata = entity.metadata or {}
    value = metadata.get("selected_candidate_key")
    return str(value) if isinstance(value, str) else None


def _decision_status_for_entity(entity_id: str, entities: list[CanonicalEntityRecord]) -> str:
    """Resolve decision status for one canonical entity identity."""
    for entity in entities:
        if entity.entity_id == entity_id:
            return entity.decision_status or "linked"
    return "linked"


def _decision_score_for_entity(entity_id: str, entities: list[CanonicalEntityRecord]) -> float | None:
    """Resolve decision score for one canonical entity identity."""
    for entity in entities:
        if entity.entity_id == entity_id:
            return entity.decision_score
    return None
