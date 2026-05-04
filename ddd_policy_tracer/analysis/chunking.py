"""Deterministic chunking service for versioned source documents."""

from __future__ import annotations

from hashlib import sha256

from ddd_policy_tracer.discovery.domain import SourceDocumentVersion

from .chunking_models import ChunkingConfig, DocumentChunk


def chunk_document_version(
    *,
    version: SourceDocumentVersion,
    config: ChunkingConfig | None = None,
) -> list[DocumentChunk]:
    """Split one document version into stable overlapping character chunks."""
    active_config = config or ChunkingConfig()
    _validate_chunking_config(active_config)

    normalized_text = version.normalized_text.strip()
    if not normalized_text:
        return []

    step = active_config.chunk_size_chars - active_config.chunk_overlap_chars
    chunks: list[DocumentChunk] = []
    index = 0
    start = 0

    while start < len(normalized_text):
        end = min(start + active_config.chunk_size_chars, len(normalized_text))
        chunk_text = normalized_text[start:end]
        if chunk_text.strip():
            chunk_id = _build_chunk_id(
                source_id=version.source_id,
                source_document_id=version.source_document_id,
                checksum=version.checksum,
                chunk_index=index,
                start_char=start,
                end_char=end,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_id=version.source_id,
                    source_document_id=version.source_document_id,
                    document_checksum=version.checksum,
                    chunk_index=index,
                    start_char=start,
                    end_char=end,
                    chunk_text=chunk_text,
                ),
            )
            index += 1

        if end >= len(normalized_text):
            break
        start += step

    return chunks


def _validate_chunking_config(config: ChunkingConfig) -> None:
    """Reject chunking settings that cannot make forward progress."""
    if config.chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be greater than zero")
    if config.chunk_overlap_chars < 0:
        raise ValueError("chunk_overlap_chars must be zero or greater")
    if config.chunk_overlap_chars >= config.chunk_size_chars:
        raise ValueError(
            "chunk_overlap_chars must be less than chunk_size_chars",
        )


def _build_chunk_id(
    *,
    source_id: str,
    source_document_id: str,
    checksum: str,
    chunk_index: int,
    start_char: int,
    end_char: int,
) -> str:
    """Create a stable identifier for one chunk of one document version."""
    raw_id = (
        f"{source_id}|{source_document_id}|{checksum}|"
        f"{chunk_index}|{start_char}|{end_char}"
    )
    digest = sha256(raw_id.encode("utf-8")).hexdigest()
    return f"chunk_{digest[:16]}"
