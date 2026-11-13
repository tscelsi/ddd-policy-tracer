from typing import Literal

import pytest
from torch import Tensor

from ddd_policy_tracer.enrichment.chunking.models import DocumentChunk
from ddd_policy_tracer.enrichment.chunking.service_layer import generate_embeddings
from ddd_policy_tracer.enrichment.embedding import AbstractEmbeddingService


def sample_chunk(*, document_checksum: str = "checksum-1") -> DocumentChunk:
    """Build one representative chunk record fixture."""
    return DocumentChunk(
        chunk_id="chunk_abc",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum=document_checksum,
        chunk_index=0,
        start_char=0,
        end_char=10,
        chunk_text="abcdefghij",
    )


class MockEmbeddingService(AbstractEmbeddingService):
    def embed_many(
        self,
        texts: list[str],
        prompt_name: Literal["query", "document"] = "document",
        return_tensor: bool = False,
    ) -> list[list[float]] | Tensor:
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.fixture
def embedding_service() -> MockEmbeddingService:
    """Build a mock embedding service for testing."""
    return MockEmbeddingService()


def test_generate_embeddings_updates_returned_chunks(
    embedding_service: AbstractEmbeddingService,
) -> None:
    """That generate_embeddings returns chunks with updated embedding vectors."""
    chunk = sample_chunk()
    embedded_chunks = generate_embeddings(chunks=[chunk], embedding_service=embedding_service)
    assert len(embedded_chunks) == 1
    assert embedded_chunks[0].embedding == [0.1, 0.2, 0.3]
