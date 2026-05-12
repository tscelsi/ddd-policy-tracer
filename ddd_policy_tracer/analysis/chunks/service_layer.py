"""Application services for chunking persisted source document versions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ddd_policy_tracer.analysis.chunks.chunking import Chunker
from ddd_policy_tracer.discovery.domain import SourceDocumentVersion

from .adapters import (
    FilesystemDocumentChunkRepository,
    SQLiteDocumentChunkRepository,
)


@dataclass(frozen=True)
class ChunkingReport:
    """Capture aggregate outcomes for one chunking execution."""

    processed_documents: int
    chunked_documents: int
    skipped_documents: int
    persisted_chunks: int


def chunk_and_persist_document_versions(
    *,
    documents: list[SourceDocumentVersion],
    chunker: Chunker,
    state_path: Path,
) -> ChunkingReport:
    """Chunk source document versions and persist new chunk records."""
    repository = _build_repository(
        state_path=state_path,
    )

    processed_documents = 0
    chunked_documents = 0
    skipped_documents = 0
    persisted_chunks = 0

    for document in documents:
        processed_documents += 1
        already_chunked = repository.has_chunks_for_document_version(
            source_id=document.source_id,
            source_document_id=document.source_document_id,
            document_checksum=document.checksum,
        )
        if already_chunked:
            skipped_documents += 1
            continue

        chunks = chunker.chunk_document_version(version=document)
        if not chunks:
            skipped_documents += 1
            continue

        persisted_chunks += repository.add_chunks(chunks)
        chunked_documents += 1

    return ChunkingReport(
        processed_documents=processed_documents,
        chunked_documents=chunked_documents,
        skipped_documents=skipped_documents,
        persisted_chunks=persisted_chunks,
    )


def _build_repository(
    *,
    state_path: Path,
) -> SQLiteDocumentChunkRepository | FilesystemDocumentChunkRepository:
    """Build the configured chunk repository adapter."""
    return FilesystemDocumentChunkRepository(state_path)
