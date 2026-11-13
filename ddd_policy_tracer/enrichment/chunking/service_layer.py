"""Application services for chunking persisted source document versions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

from ddd_policy_tracer.enrichment.chunking.chunking import get_chunker
from ddd_policy_tracer.enrichment.chunking.models import DocumentChunk
from ddd_policy_tracer.enrichment.embedding import AbstractEmbeddingService
from ddd_policy_tracer.ingestion.domain import SourceDocument

from .adapters import (
    AbstractChunkRepository,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkingReport:
    """Capture aggregate outcomes for one chunking execution."""

    processed_documents: int
    chunked_documents: int
    skipped_documents: int


async def chunk_and_persist_documents(
    *,
    repository: AbstractChunkRepository,
    documents: list[SourceDocument],
) -> ChunkingReport:
    """Chunk source document versions and persist new chunk records."""
    processed_documents = 0
    chunked_documents = 0
    skipped_documents = 0

    for document in documents:
        logger.info(f"Chunking document source_id={document.source_id} version={document.checksum}")
        chunker = get_chunker(document.source_id)
        processed_documents += 1
        chunks = chunker.chunk_document(doc=document)
        if not chunks:
            logger.warning(
                f"Document source_id={document.source_id} produced no chunks, skipping",
            )
            skipped_documents += 1
            continue
        await repository.save_bulk(chunks=chunks)
        chunked_documents += 1

    return ChunkingReport(
        processed_documents=processed_documents,
        chunked_documents=chunked_documents,
        skipped_documents=skipped_documents,
    )


def generate_embeddings(
    chunks: list[DocumentChunk],
    embedding_service: AbstractEmbeddingService,
    batch_size: int = 50,
) -> list[DocumentChunk]:
    """Generate embeddings for multiple chunks."""
    logger.info(f"Generating embeddings for {len(chunks)} chunks")
    num_batches = len(chunks) // batch_size + 1
    for i in range(0, len(chunks), batch_size):
        batch_i = i // batch_size + 1
        logger.info(f"[{batch_i}/{num_batches}] processing batch")
        batch_chunks = chunks[i : i + batch_size]
        embeddings = embedding_service.embed_many([c.chunk_text for c in batch_chunks])
        updated_batch = [
            chunk.update_embedding(cast(list[float], emb))
            for chunk, emb in zip(batch_chunks, embeddings, strict=True)
        ]
        chunks[i : i + batch_size] = updated_batch
        logger.info(
            f"[{batch_i}/{num_batches}] finished generating chunk embeddings.",
        )
    return chunks
