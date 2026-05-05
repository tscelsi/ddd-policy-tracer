"""Filesystem persistence adapter for claim candidate records."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk

from .models import ClaimCandidate
from .ports import ChunkRepository, ClaimRepository


class FilesystemChunkRepository(ChunkRepository):
    """Load document chunks from append-only JSONL state."""

    def __init__(self, state_path: Path) -> None:
        """Bind chunk repository to one JSONL file path on disk."""
        self._state_path = state_path

    def get_chunk(self, *, chunk_id: str) -> DocumentChunk | None:
        """Return one chunk by identifier or none when missing."""
        if not self._state_path.exists():
            return None

        content = self._state_path.read_text(encoding="utf-8")
        for raw_line in content.splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if payload["chunk_id"] != chunk_id:
                continue
            return DocumentChunk(
                chunk_id=payload["chunk_id"],
                source_id=payload["source_id"],
                source_document_id=payload["source_document_id"],
                document_checksum=payload["document_checksum"],
                chunk_index=payload["chunk_index"],
                start_char=payload["start_char"],
                end_char=payload["end_char"],
                chunk_text=payload["chunk_text"],
            )
        return None


class FilesystemClaimRepository(ClaimRepository):
    """Store and retrieve claim candidates from append-only JSONL state."""

    def __init__(self, state_path: Path) -> None:
        """Bind repository to one JSONL file path on disk."""
        self._state_path = state_path
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    def add_claims(self, claims: list[ClaimCandidate]) -> int:
        """Append claim records to JSONL and return inserted count."""
        if not claims:
            return 0

        existing_keys = {
            _claim_dedup_key(claim)
            for claim in self._read_all()
        }
        to_insert: list[ClaimCandidate] = []
        for claim in claims:
            dedup_key = _claim_dedup_key(claim)
            if dedup_key in existing_keys:
                continue
            existing_keys.add(dedup_key)
            to_insert.append(claim)

        if not to_insert:
            return 0

        with self._state_path.open("a", encoding="utf-8") as handle:
            for claim in to_insert:
                record = {
                    "claim_id": claim.claim_id,
                    "chunk_id": claim.chunk_id,
                    "source_id": claim.source_id,
                    "source_document_id": claim.source_document_id,
                    "document_checksum": claim.document_checksum,
                    "start_char": claim.start_char,
                    "end_char": claim.end_char,
                    "evidence_text": claim.evidence_text,
                    "normalized_claim_text": claim.normalized_claim_text,
                    "confidence": claim.confidence,
                    "claim_type": claim.claim_type,
                    "extractor_version": claim.extractor_version,
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return len(to_insert)

    def list_claims(self, *, chunk_id: str | None = None) -> list[ClaimCandidate]:
        """List persisted claim records, optionally filtered by chunk identity."""
        claims = self._read_all()
        if chunk_id is None:
            return claims
        return [claim for claim in claims if claim.chunk_id == chunk_id]

    def _read_all(self) -> list[ClaimCandidate]:
        """Read all claim records from JSONL persistence state."""
        if not self._state_path.exists():
            return []

        claims: list[ClaimCandidate] = []
        content = self._state_path.read_text(encoding="utf-8")
        for raw_line in content.splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            claims.append(
                ClaimCandidate(
                    claim_id=payload["claim_id"],
                    chunk_id=payload["chunk_id"],
                    source_id=payload["source_id"],
                    source_document_id=payload["source_document_id"],
                    document_checksum=payload["document_checksum"],
                    start_char=payload["start_char"],
                    end_char=payload["end_char"],
                    evidence_text=payload["evidence_text"],
                    normalized_claim_text=payload["normalized_claim_text"],
                    confidence=payload["confidence"],
                    claim_type=payload["claim_type"],
                    extractor_version=payload["extractor_version"],
                ),
            )
        return claims


def _claim_dedup_key(claim: ClaimCandidate) -> tuple[str, str, str, int, int, str]:
    """Build tuple key used for idempotent claim persistence behavior."""
    return (
        claim.chunk_id,
        claim.source_id,
        claim.document_checksum,
        claim.start_char,
        claim.end_char,
        claim.extractor_version,
    )
