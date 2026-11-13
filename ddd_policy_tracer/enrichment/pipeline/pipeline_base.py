"""Pipeline components for enriching document chunks with metadata and embeddings."""

import logging
from collections.abc import Generator

from ddd_policy_tracer.enrichment.chunking.adapters import AbstractChunkRepository
from ddd_policy_tracer.enrichment.chunking.models import DocumentChunk
from ddd_policy_tracer.enrichment.pipeline.layers import BaseLayer

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrate a sequence of enrichment layers to process document chunks."""

    def __init__(
        self,
        layers: list[BaseLayer],
        store: AbstractChunkRepository | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Initialize the pipeline with a sequence of enrichment layers."""
        self._layers = layers
        self._store = store
        self._batch_size = batch_size

    def _get_embeddings_hashes(self, chunks: list[DocumentChunk]) -> dict[str, int]:
        """Generate a hash for each chunk based on its embeddings."""
        return {
            chunk.chunk_id: hash(tuple(chunk.embedding) if chunk.embedding else None)
            for chunk in chunks
        }

    def _get_object_hashes(self, chunks: list[DocumentChunk]) -> dict[str, int]:
        """Generate a hash for each chunk object (excluding embeddings)."""
        chunks_no_embeddings = [chunk.update_embedding(None) for chunk in chunks]
        return {chunk.chunk_id: hash(chunk.serialize()) for chunk in chunks_no_embeddings}

    async def run(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Run the input chunks through each layer in sequence."""
        returned_chunks: list[DocumentChunk] = []
        for batch, start_idx, end_idx in self._batch_generator(
            chunks=chunks,
        ):
            logger.info(f"Processing chunks [{start_idx}-{end_idx}]/{len(chunks)}")
            for layer in self._layers:
                logger.info(f"Running layer: {layer.__class__.__name__}")
                _embeddings_pre_hash = self._get_embeddings_hashes(batch)
                _chunks_pre_hash = self._get_object_hashes(batch)
                batch = layer.call(batch)
                if not batch:
                    logger.warning(
                        "Layer %s returned no chunks, skipping remaining layers",
                        layer.__class__.__name__,
                    )
                    break
                to_update_embeddings: list[DocumentChunk] = []
                to_update_objects: list[DocumentChunk] = []
                _embeddings_post_hash = self._get_embeddings_hashes(batch)
                _chunks_post_hash = self._get_object_hashes(batch)
                for updated_chunk in batch:
                    if (
                        _embeddings_pre_hash[updated_chunk.chunk_id]
                        != _embeddings_post_hash[updated_chunk.chunk_id]
                    ):
                        to_update_embeddings.append(updated_chunk)
                    if (
                        _chunks_pre_hash[updated_chunk.chunk_id]
                        != _chunks_post_hash[updated_chunk.chunk_id]
                    ):
                        to_update_objects.append(updated_chunk)
                if self._store:
                    if to_update_embeddings:
                        logger.info(f"Layer {layer.__class__.__name__} modified embeddings")
                        await self._store.save_embeddings(chunks=to_update_embeddings)
                    if to_update_objects:
                        logger.info(f"Layer {layer.__class__.__name__} modified chunk objects")
                        await self._store.save_bulk(chunks=to_update_objects)
                logger.info(f"Layer {layer.__class__.__name__} output {len(batch)} chunks")
            returned_chunks.extend(batch)
        return returned_chunks

    def _batch_generator(
        self,
        chunks: list[DocumentChunk],
    ) -> Generator[tuple[list[DocumentChunk], int, int], None, None]:
        """Yield successive batches of chunks."""
        total_chunks = len(chunks)
        if self._batch_size is None:
            yield (chunks, 0, total_chunks)
        else:
            for i in range(0, len(chunks), self._batch_size):
                yield (chunks[i : i + self._batch_size], i, min(i + self._batch_size, total_chunks))
