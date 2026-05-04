"""Typed records for deterministic document chunking outputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    """Configure deterministic chunk size and overlap boundaries."""

    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 200


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
