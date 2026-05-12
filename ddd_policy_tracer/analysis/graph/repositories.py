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
            record = ClaimRecord(
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
                claim_type=payload["claim_type"]
                if payload["claim_type"] is None
                else str(payload["claim_type"]),
                extractor_version=str(payload["extractor_version"]),
            )
            if source_id is not None and record.source_id != source_id:
                continue
            claims.append(record)
        return claims


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
