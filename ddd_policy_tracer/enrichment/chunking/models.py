"""Typed records for deterministic document chunking outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass(frozen=True)
class ChunkingConfig:
    """Configure deterministic chunk size and overlap boundaries."""

    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200


class Topic(TypedDict):
    """Represent one topic label with confidence score."""

    topic: str
    confidence: float


@dataclass(frozen=True)
class DocumentChunk:
    """Represent one version-bound text chunk with traceable offsets."""

    chunk_id: str
    source_id: str
    source_document_id: str
    document_checksum: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_text: str
    topics: list[Topic] | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None

    def update_embedding(self, embedding: list[float] | None) -> DocumentChunk:
        """Update the chunk with a new embedding vector."""
        return DocumentChunk(
            chunk_id=self.chunk_id,
            source_id=self.source_id,
            source_document_id=self.source_document_id,
            document_checksum=self.document_checksum,
            chunk_index=self.chunk_index,
            start_char=self.start_char,
            end_char=self.end_char,
            chunk_text=self.chunk_text,
            topics=self.topics,
            embedding=embedding,
            metadata=self.metadata,
        )

    def update_topics(self, topics: list[Topic]) -> DocumentChunk:
        """Update the chunk with new topic labels."""
        return DocumentChunk(
            chunk_id=self.chunk_id,
            source_id=self.source_id,
            source_document_id=self.source_document_id,
            document_checksum=self.document_checksum,
            chunk_index=self.chunk_index,
            start_char=self.start_char,
            end_char=self.end_char,
            chunk_text=self.chunk_text,
            topics=topics,
            embedding=self.embedding,
            metadata=self.metadata,
        )

    def serialize(self) -> str:
        """Convert the DocumentChunk to a JSON string for storage or transmission."""
        return json.dumps(
            {
                "chunk_id": self.chunk_id,
                "source_id": self.source_id,
                "source_document_id": self.source_document_id,
                "document_checksum": self.document_checksum,
                "chunk_index": self.chunk_index,
                "start_char": self.start_char,
                "end_char": self.end_char,
                "chunk_text": self.chunk_text,
                "topics": self.topics,
                "embedding": self.embedding,
                "metadata": self.metadata,
            },
            default=str,
        )
