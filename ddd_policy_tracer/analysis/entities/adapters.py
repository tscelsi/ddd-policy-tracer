"""Filesystem persistence adapter for entity mention records."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk

from .models import EntityMention
from .ports import ChunkRepository, EntityRepository


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

    def list_chunk_ids(self) -> list[str]:
        """Return all unique chunk ids from JSONL in first-seen order."""
        if not self._state_path.exists():
            return []

        seen: set[str] = set()
        chunk_ids: list[str] = []
        content = self._state_path.read_text(encoding="utf-8")
        for raw_line in content.splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                continue
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
        return chunk_ids


class FilesystemEntityRepository(EntityRepository):
    """Store and retrieve entity mentions from append-only JSONL state."""

    def __init__(self, state_path: Path) -> None:
        """Bind repository to one JSONL file path on disk."""
        self._state_path = state_path
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    def add_entities(self, entities: list[EntityMention]) -> int:
        """Append entity records to JSONL and return inserted count."""
        if not entities:
            return 0

        existing_keys = {
            _entity_dedup_key(entity)
            for entity in self._read_all()
        }
        to_insert: list[EntityMention] = []
        for entity in entities:
            dedup_key = _entity_dedup_key(entity)
            if dedup_key in existing_keys:
                continue
            existing_keys.add(dedup_key)
            to_insert.append(entity)

        if not to_insert:
            return 0

        with self._state_path.open("a", encoding="utf-8") as handle:
            for entity in to_insert:
                record = {
                    "entity_id": entity.entity_id,
                    "chunk_id": entity.chunk_id,
                    "source_id": entity.source_id,
                    "source_document_id": entity.source_document_id,
                    "document_checksum": entity.document_checksum,
                    "start_char": entity.start_char,
                    "end_char": entity.end_char,
                    "mention_text": entity.mention_text,
                    "normalized_mention_text": entity.normalized_mention_text,
                    "entity_type": entity.entity_type,
                    "confidence": entity.confidence,
                    "extractor_version": entity.extractor_version,
                    "canonical_entity_key": entity.canonical_entity_key,
                    "metadata": entity.metadata,
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return len(to_insert)

    def list_entities(self, *, chunk_id: str | None = None) -> list[EntityMention]:
        """List persisted entity records, optionally filtered by chunk identity."""
        entities = self._read_all()
        if chunk_id is None:
            return entities
        return [entity for entity in entities if entity.chunk_id == chunk_id]

    def _read_all(self) -> list[EntityMention]:
        """Read all entity records from JSONL persistence state."""
        if not self._state_path.exists():
            return []

        entities: list[EntityMention] = []
        content = self._state_path.read_text(encoding="utf-8")
        for raw_line in content.splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            entities.append(
                EntityMention(
                    entity_id=payload["entity_id"],
                    chunk_id=payload["chunk_id"],
                    source_id=payload["source_id"],
                    source_document_id=payload["source_document_id"],
                    document_checksum=payload["document_checksum"],
                    start_char=payload["start_char"],
                    end_char=payload["end_char"],
                    mention_text=payload["mention_text"],
                    normalized_mention_text=payload["normalized_mention_text"],
                    entity_type=payload["entity_type"],
                    confidence=payload["confidence"],
                    extractor_version=payload["extractor_version"],
                    canonical_entity_key=payload["canonical_entity_key"],
                    metadata=payload["metadata"],
                ),
            )
        return entities


def _entity_dedup_key(
    entity: EntityMention,
) -> tuple[str, str, str, int, int, str, str]:
    """Build tuple key used for idempotent entity persistence behavior."""
    return (
        entity.chunk_id,
        entity.source_id,
        entity.document_checksum,
        entity.start_char,
        entity.end_char,
        entity.entity_type,
        entity.extractor_version,
    )
