"""Service layer for enrichment operations on document chunks, such as topic classification."""

from ddd_policy_tracer.enrichment.chunking.adapters import AbstractChunkRepository
from ddd_policy_tracer.enrichment.chunking.models import DocumentChunk
from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.enrichment.pipeline import Pipeline, layers


async def run_full_pipeline_on_chunks(
    chunks: list[DocumentChunk],
    chunk_repo: AbstractChunkRepository,
    embedding_service: EmbeddingService,
    batch_size: int | None = None,
) -> list[DocumentChunk]:
    """Run the full enrichment pipeline to add embeddings and topic labels to chunks."""
    pipeline = Pipeline(
        layers=[
            # layers.Filter(
            #     lambda chunk: len(chunk.chunk_text) > 500 and len(chunk.chunk_text) <= 2500,
            # ),  # Keep chunks that are big enough.
            layers.Filter(
                lambda chunk: chunk.topics is None or chunk.embedding is None,
            ),
            layers.Embed(embedding_service=embedding_service),
            layers.ClassifyTopics(),
        ],
        store=chunk_repo,
        batch_size=batch_size,
    )
    enriched_chunks = await pipeline.run(chunks)
    return enriched_chunks


async def run_full_pipeline(
    chunk_repo: AbstractChunkRepository,
    embedding_service: EmbeddingService,
    batch_size: int | None = None,
) -> list[DocumentChunk]:
    """Run the full enrichment pipeline to add embeddings and topic labels to chunks."""
    chunks = await chunk_repo.list_chunks()
    pipeline = Pipeline(
        layers=[
            layers.Filter(
                lambda chunk: len(chunk.chunk_text) > 500 and len(chunk.chunk_text) <= 2500,
            ),  # Keep chunks that are big enough.
            layers.Embed(embedding_service=embedding_service),
            layers.ClassifyTopics(),
        ],
        store=chunk_repo,
        batch_size=batch_size,
    )
    enriched_chunks = await pipeline.run(chunks)
    return enriched_chunks


async def run_embeddings_on_chunks(
    chunks: list[DocumentChunk],
    chunk_repo: AbstractChunkRepository,
    embedding_service: EmbeddingService,
    batch_size: int | None = None,
) -> list[DocumentChunk]:
    """Add vector embeddings to each chunk using the provided embedding service."""
    pipeline = Pipeline(
        layers=[
            layers.Filter(
                lambda chunk: len(chunk.chunk_text) > 500 and chunk.embedding is None,
            ),  # Keep chunks that are big enough, and don't already have embeddings.
            layers.Embed(embedding_service=embedding_service),
        ],
        store=chunk_repo,
        batch_size=batch_size,
    )
    chunks = await pipeline.run(chunks)
    return chunks


async def run_topic_classification_on_chunks(
    chunks: list[DocumentChunk],
    chunk_repo: AbstractChunkRepository,
    batch_size: int | None = None,
) -> list[DocumentChunk]:
    """Add topic labels to each chunk using a multi-label classifier."""
    pipeline = Pipeline(
        layers=[
            layers.Filter(
                lambda chunk: (
                    len(chunk.chunk_text) > 500
                    and len(chunk.chunk_text) <= 2500
                    and not chunk.topics
                ),
            ),  # Keep chunks that are right size, and haven't been classified already.
            layers.ClassifyTopics(),
        ],
        store=chunk_repo,
        batch_size=batch_size,
    )
    chunks = await pipeline.run(chunks)
    return chunks
