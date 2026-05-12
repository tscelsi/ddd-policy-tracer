"""Repository ports and JSONL adapters for graph materialization inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClaimRecord:
    """Represent one persisted claim row consumed by graph materialization."""

    claim_id: str
    chunk_id: str
    source_id: str
    source_document_id: str
    document_checksum: str
    start_char: int
    end_char: int
    evidence_text: str
    normalized_claim_text: str
    confidence: float
    claim_type: str | None
    extractor_version: str


@dataclass(frozen=True)
class EntityRecord:
    """Represent one persisted entity row consumed by graph materialization."""

    entity_id: str
    chunk_id: str
    source_id: str
    source_document_id: str
    document_checksum: str
    start_char: int
    end_char: int
    mention_text: str
    normalized_mention_text: str
    entity_type: str
    confidence: float
    extractor_version: str
    canonical_entity_key: str | None


class EntityRepository:
    """Define entity-read behavior required by graph materialization."""

    def list_entities(self, *, source_id: str | None = None) -> list[EntityRecord]:
        """Load persisted entity rows, optionally filtered by source."""
        raise NotImplementedError


class ClaimRepository:
    """Define claim-read behavior required by graph materialization."""

    def list_claims(self, *, source_id: str | None = None) -> list[ClaimRecord]:
        """Load persisted claim rows, optionally filtered by source."""
        raise NotImplementedError


class JsonlClaimRepository(ClaimRepository):
    """Load claim rows from append-only JSONL persistence state."""

    def __init__(self, *, path: Path) -> None:
        """Bind claim repository to one JSONL artifact path."""
        self._path = path

    def list_claims(self, *, source_id: str | None = None) -> list[ClaimRecord]:
        """Load claims from JSONL and optionally filter by source ID."""
        claims: list[ClaimRecord] = []
        for payload in _read_json_objects(self._path):
            try:
                record = _payload_to_claim(payload)
            except ValueError:
                continue
            if source_id is not None and record.source_id != source_id:
                continue
            claims.append(record)
        return claims


class JsonlEntityRepository(EntityRepository):
    """Load entity rows from append-only JSONL persistence state."""

    def __init__(self, *, path: Path) -> None:
        """Bind entity repository to one JSONL artifact path."""
        self._path = path

    def list_entities(self, *, source_id: str | None = None) -> list[EntityRecord]:
        """Load entities from JSONL and optionally filter by source ID."""
        entities: list[EntityRecord] = []
        for payload in _read_json_objects(self._path):
            try:
                record = _payload_to_entity(payload)
            except ValueError:
                continue
            if source_id is not None and record.source_id != source_id:
                continue
            entities.append(record)
        return entities


def _read_json_objects(path: Path) -> list[dict[str, object]]:
    """Read non-empty JSON object rows from one JSONL path."""
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _payload_to_claim(payload: dict[str, object]) -> ClaimRecord:
    """Translate one JSON object payload into a claim record."""
    _require_fields(
        payload=payload,
        required_fields=(
            "claim_id",
            "chunk_id",
            "source_id",
            "source_document_id",
            "document_checksum",
            "start_char",
            "end_char",
            "evidence_text",
            "normalized_claim_text",
            "confidence",
            "extractor_version",
        ),
    )
    claim_type_value = payload.get("claim_type")
    claim_type = claim_type_value if claim_type_value is None else str(claim_type_value)
    return ClaimRecord(
        claim_id=str(payload["claim_id"]),
        chunk_id=str(payload["chunk_id"]),
        source_id=str(payload["source_id"]),
        source_document_id=str(payload["source_document_id"]),
        document_checksum=str(payload["document_checksum"]),
        start_char=int(payload["start_char"]),
        end_char=int(payload["end_char"]),
        evidence_text=str(payload["evidence_text"]),
        normalized_claim_text=str(payload["normalized_claim_text"]),
        confidence=float(payload["confidence"]),
        claim_type=claim_type,
        extractor_version=str(payload["extractor_version"]),
    )


def _payload_to_entity(payload: dict[str, object]) -> EntityRecord:
    """Translate one JSON object payload into an entity record."""
    _require_fields(
        payload=payload,
        required_fields=(
            "entity_id",
            "chunk_id",
            "source_id",
            "source_document_id",
            "document_checksum",
            "start_char",
            "end_char",
            "mention_text",
            "normalized_mention_text",
            "entity_type",
            "confidence",
            "extractor_version",
        ),
    )
    canonical_key = payload.get("canonical_entity_key")
    return EntityRecord(
        entity_id=str(payload["entity_id"]),
        chunk_id=str(payload["chunk_id"]),
        source_id=str(payload["source_id"]),
        source_document_id=str(payload["source_document_id"]),
        document_checksum=str(payload["document_checksum"]),
        start_char=int(payload["start_char"]),
        end_char=int(payload["end_char"]),
        mention_text=str(payload["mention_text"]),
        normalized_mention_text=str(payload["normalized_mention_text"]),
        entity_type=str(payload["entity_type"]),
        confidence=float(payload["confidence"]),
        extractor_version=str(payload["extractor_version"]),
        canonical_entity_key=canonical_key if canonical_key is None else str(canonical_key),
    )


def _require_fields(*, payload: dict[str, object], required_fields: tuple[str, ...]) -> None:
    """Raise ValueError when one JSON payload misses required keys."""
    missing = [field for field in required_fields if field not in payload]
    if missing:
        missing_fields = ", ".join(sorted(missing))
        raise ValueError(f"missing required fields: {missing_fields}")
