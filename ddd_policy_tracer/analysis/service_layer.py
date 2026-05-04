"""Application services for chunking persisted source document versions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ddd_policy_tracer.discovery.domain import SourceDocumentVersion

from .adapters import (
    FilesystemDocumentChunkRepository,
    SQLiteDocumentChunkRepository,
)
from .chunking import chunk_document_version
from .chunking_models import ChunkingConfig


@dataclass(frozen=True)
class ChunkingReport:
    """Capture aggregate outcomes for one chunking execution."""

    processed_documents: int
    chunked_documents: int
    skipped_documents: int
    persisted_chunks: int


def chunk_and_persist_document_versions(
    *,
    versions: list[SourceDocumentVersion],
    state_path: Path,
    repository_backend: Literal["sqlite", "filesystem"] = "sqlite",
    config: ChunkingConfig | None = None,
) -> ChunkingReport:
    """Chunk source document versions and persist new chunk records."""
    repository = _build_repository(
        state_path=state_path,
        repository_backend=repository_backend,
    )
    active_config = config or ChunkingConfig()

    processed_documents = 0
    chunked_documents = 0
    skipped_documents = 0
    persisted_chunks = 0

    for version in versions:
        processed_documents += 1
        already_chunked = repository.has_chunks_for_document_version(
            source_id=version.source_id,
            source_document_id=version.source_document_id,
            document_checksum=version.checksum,
        )
        if already_chunked:
            skipped_documents += 1
            continue

        chunks = chunk_document_version(version=version, config=active_config)
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
    repository_backend: Literal["sqlite", "filesystem"],
) -> SQLiteDocumentChunkRepository | FilesystemDocumentChunkRepository:
    """Build the configured chunk repository adapter."""
    if repository_backend == "filesystem":
        return FilesystemDocumentChunkRepository(state_path)
    return SQLiteDocumentChunkRepository(state_path)
