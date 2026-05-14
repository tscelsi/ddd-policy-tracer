"""JSONL adapters for canonicalization stage inputs and outputs."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Literal, cast

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


class JsonlEntitySourceRepository:
    """Load extracted entity mentions from JSONL state."""

    def __init__(self, *, path: Path) -> None:
        """Bind source repository to one entities JSONL path."""
        self._path = path

    def list_entities(self) -> list[EntityMention]:
        """Load all source entity rows from JSONL state."""
        rows: list[EntityMention] = []
        for payload in _read_json_objects(self._path):
            rows.append(
                EntityMention(
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
                    canonical_entity_key=(
                        None
                        if payload.get("canonical_entity_key") is None
                        else str(payload.get("canonical_entity_key"))
                    ),
                    metadata=(
                        None
                        if payload.get("metadata") is None
                        else dict(payload.get("metadata", {}))
                    ),
                ),
            )
        return rows

    def artifact_hash(self) -> str:
        """Return deterministic hash for repository backing JSONL path."""
        return _artifact_hash(self._path)


class JsonlClaimSourceRepository:
    """Load extracted claim candidates from JSONL state."""

    def __init__(self, *, path: Path) -> None:
        """Bind source repository to one claims JSONL path."""
        self._path = path

    def list_claims(self) -> list[ClaimCandidate]:
        """Load all source claim rows from JSONL state."""
        rows: list[ClaimCandidate] = []
        for payload in _read_json_objects(self._path):
            rows.append(
                ClaimCandidate(
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
                    claim_type=(
                        None
                        if payload.get("claim_type") is None
                        else str(payload.get("claim_type"))
                    ),
                    extractor_version=str(payload["extractor_version"]),
                ),
            )
        return rows

    def artifact_hash(self) -> str:
        """Return deterministic hash for repository backing JSONL path."""
        return _artifact_hash(self._path)


class JsonlCanonicalEntityRepository:
    """Persist and load canonical entity rows in JSONL format."""

    def __init__(self, *, path: Path) -> None:
        """Bind canonical repository to one entity canonical JSONL path."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def replace_all(self, *, rows: list[CanonicalEntityRecord]) -> None:
        """Replace canonical entity artifact with all provided rows."""
        with self._path.open("w", encoding="utf-8") as handle:
            for row in rows:
                payload = _canonical_entity_to_payload(row)
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def list_entities(self) -> list[CanonicalEntityRecord]:
        """Load canonical entity rows from JSONL state."""
        rows: list[CanonicalEntityRecord] = []
        for payload in _read_json_objects(self._path):
            rows.append(_payload_to_canonical_entity(payload))
        return rows

    def artifact_hash(self) -> str:
        """Return deterministic hash for canonical entity artifact path."""
        return _artifact_hash(self._path)


class JsonlCanonicalClaimRepository:
    """Persist canonical claim rows in JSONL format."""

    def __init__(self, *, path: Path) -> None:
        """Bind canonical repository to one claim canonical JSONL path."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def replace_all(self, *, rows: list[CanonicalClaimRecord]) -> None:
        """Replace canonical claim artifact with all provided rows."""
        with self._path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_canonical_claim_to_payload(row), ensure_ascii=True) + "\n")

    def artifact_hash(self) -> str:
        """Return deterministic hash for canonical claim artifact path."""
        return _artifact_hash(self._path)

    def list_claims(self) -> list[CanonicalClaimRecord]:
        """Load canonical claim rows from JSONL state for tests."""
        rows: list[CanonicalClaimRecord] = []
        for payload in _read_json_objects(self._path):
            rows.append(_payload_to_canonical_claim(payload))
        return rows


def _canonical_entity_to_payload(row: CanonicalEntityRecord) -> dict[str, object]:
    """Serialize canonical entity row to JSON-serializable payload."""
    return {
        "entity_id": row.entity_id,
        "chunk_id": row.chunk_id,
        "source_id": row.source_id,
        "source_document_id": row.source_document_id,
        "document_checksum": row.document_checksum,
        "start_char": row.start_char,
        "end_char": row.end_char,
        "mention_text": row.mention_text,
        "normalized_mention_text": row.normalized_mention_text,
        "entity_type": row.entity_type,
        "confidence": row.confidence,
        "extractor_version": row.extractor_version,
        "canonical_name": row.canonical_name,
        "canonical_entity_key": row.canonical_entity_key,
        "decision_status": row.decision_status,
        "decision_score": row.decision_score,
        "selected_candidate_key": row.selected_candidate_key,
        "canonicalization": _canonicalization_metadata_to_payload(row.canonicalization),
    }


def _payload_to_canonical_entity(payload: dict[str, object]) -> CanonicalEntityRecord:
    """Deserialize one canonical entity row payload."""
    canonicalization_raw = payload.get("canonicalization")
    canonicalization_payload = (
        canonicalization_raw if isinstance(canonicalization_raw, dict) else {}
    )
    return CanonicalEntityRecord(
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
        canonical_name=str(payload["canonical_name"]),
        canonical_entity_key=str(payload["canonical_entity_key"]),
        canonicalization=_payload_to_canonicalization_metadata(canonicalization_payload),
        decision_status=(
            str(payload.get("decision_status"))
            if isinstance(payload.get("decision_status"), str)
            else None
        ),
        decision_score=(
            float(payload.get("decision_score"))
            if isinstance(payload.get("decision_score"), (int, float))
            else None
        ),
        selected_candidate_key=(
            str(payload.get("selected_candidate_key"))
            if isinstance(payload.get("selected_candidate_key"), str)
            else None
        ),
    )


def _canonical_claim_to_payload(row: CanonicalClaimRecord) -> dict[str, object]:
    """Serialize canonical claim row to JSON-serializable payload."""
    return {
        "claim_id": row.claim_id,
        "chunk_id": row.chunk_id,
        "source_id": row.source_id,
        "source_document_id": row.source_document_id,
        "document_checksum": row.document_checksum,
        "start_char": row.start_char,
        "end_char": row.end_char,
        "evidence_text": row.evidence_text,
        "normalized_claim_text": row.normalized_claim_text,
        "confidence": row.confidence,
        "claim_type": row.claim_type,
        "extractor_version": row.extractor_version,
        "linked_entities": [_linked_entity_to_payload(linked) for linked in row.linked_entities],
        "pending_entity_links": [
            _pending_entity_link_to_payload(pending)
            for pending in row.pending_entity_links
        ],
        "canonicalization": _canonicalization_metadata_to_payload(row.canonicalization),
    }


def _payload_to_canonical_claim(payload: dict[str, object]) -> CanonicalClaimRecord:
    """Deserialize one canonical claim payload from JSONL storage."""
    canonicalization_raw = payload.get("canonicalization")
    canonicalization_payload = canonicalization_raw if isinstance(canonicalization_raw, dict) else {}
    linked_payload = payload.get("linked_entities")
    linked_entities = [
        linked_entity_from_payload(entry)
        for entry in linked_payload
        if isinstance(entry, dict)
    ] if isinstance(linked_payload, list) else []
    pending_payload = payload.get("pending_entity_links")
    pending_links = [
        _pending_entity_from_payload(entry)
        for entry in pending_payload
        if isinstance(entry, dict)
    ] if isinstance(pending_payload, list) else []
    return CanonicalClaimRecord(
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
        claim_type=(
            None
            if payload.get("claim_type") is None
            else str(payload.get("claim_type"))
        ),
        extractor_version=str(payload["extractor_version"]),
        linked_entities=linked_entities,
        pending_entity_links=pending_links,
        canonicalization=_payload_to_canonicalization_metadata(canonicalization_payload),
    )


def _pending_entity_link_to_payload(pending: PendingEntityLinkRecord) -> dict[str, object]:
    """Serialize one unresolved pending entity link payload."""
    payload: dict[str, object] = {
        "canonical_entity_key": pending.canonical_entity_key,
        "entity_type": pending.entity_type,
        "canonical_name": pending.canonical_name,
        "decision_status": pending.decision_status,
        "decision_score": pending.decision_score,
        "entity_id": pending.entity_id,
    }
    if pending.evidence is not None:
        payload["evidence"] = {
            "claim_span": [pending.evidence.claim_span[0], pending.evidence.claim_span[1]],
            "entity_span": [pending.evidence.entity_span[0], pending.evidence.entity_span[1]],
            "matched_text": pending.evidence.matched_text,
        }
    return payload


def _linked_entity_to_payload(linked: LinkedEntityRecord) -> dict[str, object]:
    """Serialize one linked entity row attached to a canonical claim."""
    payload: dict[str, object] = {
        "canonical_entity_key": linked.canonical_entity_key,
        "entity_type": linked.entity_type,
        "canonical_name": linked.canonical_name,
        "link_method": linked.link_method,
        "entity_id": linked.entity_id,
    }
    if linked.evidence is not None:
        payload["evidence"] = {
            "claim_span": [linked.evidence.claim_span[0], linked.evidence.claim_span[1]],
            "entity_span": [linked.evidence.entity_span[0], linked.evidence.entity_span[1]],
            "matched_text": linked.evidence.matched_text,
        }
    return payload


def _pending_entity_from_payload(payload: dict[str, object]) -> PendingEntityLinkRecord:
    """Deserialize one unresolved pending entity payload."""
    evidence_payload_raw = payload.get("evidence")
    evidence_payload = evidence_payload_raw if isinstance(evidence_payload_raw, dict) else None
    evidence = None
    if evidence_payload is not None:
        claim_span_raw = evidence_payload.get("claim_span", [0, 0])
        entity_span_raw = evidence_payload.get("entity_span", [0, 0])
        evidence = LinkedEntityEvidence(
            claim_span=(int(claim_span_raw[0]), int(claim_span_raw[1])),
            entity_span=(int(entity_span_raw[0]), int(entity_span_raw[1])),
            matched_text=(
                None
                if evidence_payload.get("matched_text") is None
                else str(evidence_payload.get("matched_text"))
            ),
        )
    decision_score_raw = payload.get("decision_score")
    return PendingEntityLinkRecord(
        canonical_entity_key=str(payload["canonical_entity_key"]),
        entity_type=str(payload["entity_type"]),
        canonical_name=str(payload["canonical_name"]),
        decision_status=str(payload["decision_status"]),
        decision_score=(
            float(decision_score_raw)
            if isinstance(decision_score_raw, (int, float))
            else None
        ),
        entity_id=str(payload["entity_id"]),
        evidence=evidence,
    )


def _canonicalization_metadata_to_payload(metadata: CanonicalizationMetadata) -> dict[str, object]:
    """Serialize canonicalization metadata embedded in canonical rows."""
    return {
        "stage": metadata.stage,
        "canonicalizer_version": metadata.canonicalizer_version,
        "generated_at": metadata.generated_at,
        "input_artifact_hash": metadata.input_artifact_hash,
    }


def _payload_to_canonicalization_metadata(payload: dict[str, object]) -> CanonicalizationMetadata:
    """Deserialize canonicalization metadata from JSON payload."""
    raw_stage = str(payload.get("stage", "entity"))
    stage: Literal["entity", "claim"]
    if raw_stage == "claim":
        stage = "claim"
    else:
        stage = "entity"
    return CanonicalizationMetadata(
        stage=stage,
        canonicalizer_version=str(payload.get("canonicalizer_version", "unknown")),
        generated_at=str(payload.get("generated_at", "")),
        input_artifact_hash=str(payload.get("input_artifact_hash", "")),
    )


def _read_json_objects(path: Path) -> list[dict[str, object]]:
    """Read all JSON object rows from one JSONL path."""
    if not path.exists():
        return []

    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _artifact_hash(path: Path) -> str:
    """Hash one JSONL artifact deterministically by file content."""
    digest = sha256()
    if not path.exists():
        digest.update(b"")
        return digest.hexdigest()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def rows_to_staleness_fingerprint(
    *,
    rows: list[CanonicalEntityRecord],
) -> tuple[set[str], set[str], str]:
    """Build source/checksum/version fingerprint from canonical entity rows."""
    source_ids = {row.source_id for row in rows}
    checksums = {row.document_checksum for row in rows}
    versions = {row.canonicalization.canonicalizer_version for row in rows}
    version = ""
    if len(versions) == 1:
        version = next(iter(versions))
    return source_ids, checksums, version


def linked_entity_from_payload(payload: dict[str, object]) -> LinkedEntityRecord:
    """Deserialize one linked entity payload for compatibility tests."""
    evidence_payload_raw = payload.get("evidence")
    evidence_payload = evidence_payload_raw if isinstance(evidence_payload_raw, dict) else None
    evidence = None
    if evidence_payload is not None:
        claim_span_raw = evidence_payload.get("claim_span", [0, 0])
        entity_span_raw = evidence_payload.get("entity_span", [0, 0])
        claim_span = (
            int(claim_span_raw[0]),
            int(claim_span_raw[1]),
        )
        entity_span = (
            int(entity_span_raw[0]),
            int(entity_span_raw[1]),
        )
        evidence = LinkedEntityEvidence(
            claim_span=claim_span,
            entity_span=entity_span,
            matched_text=(
                None
                if evidence_payload.get("matched_text") is None
                else str(evidence_payload.get("matched_text"))
            ),
        )
    raw_link_method = str(payload.get("link_method", "text_match_fallback"))
    link_method: Literal["span_overlap", "text_match_fallback"]
    if raw_link_method == "span_overlap":
        link_method = "span_overlap"
    else:
        link_method = "text_match_fallback"
    return LinkedEntityRecord(
        canonical_entity_key=str(payload["canonical_entity_key"]),
        entity_type=str(payload["entity_type"]),
        canonical_name=str(payload["canonical_name"]),
        link_method=cast(Literal["span_overlap", "text_match_fallback"], link_method),
        entity_id=str(payload["entity_id"]),
        evidence=evidence,
    )
