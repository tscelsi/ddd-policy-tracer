"""Use cases for finding parliamentary speech evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from torch import Tensor

from ddd_policy_tracer.retrieval.filtering.metadata_filters import MetadataFilter


class RetrievedChunk(Protocol):
    """Expose the evidence fields used to answer a retrieval query."""

    @property
    def chunk_id(self) -> str:
        """Return the stable identity of the evidence chunk."""
        ...

    @property
    def chunk_text(self) -> str:
        """Return the speech passage stored in the evidence chunk."""
        ...

    @property
    def source_document_id(self) -> str:
        """Return the stable identity of the source speech document."""
        ...

    @property
    def metadata(self) -> str | None:
        """Return the source metadata stored with the evidence chunk."""
        ...


class QueryEmbedder(Protocol):
    """Port for embedding a user's question for semantic retrieval."""

    def embed_one(
        self,
        text: str,
        prompt_name: Literal["query", "document"] = "document",
        return_tensor: bool = False,
    ) -> list[float] | Tensor:
        """Return a vector representation of the question."""
        ...


class HybridChunkRetriever(Protocol):
    """Port for retrieving ranked chunk evidence from a corpus."""

    def search_hybrid(
        self,
        *,
        query: str,
        query_embedding: Sequence[float],
        limit: int = 10,
        candidate_limit: int | None = None,
        filters: MetadataFilter | None = None,
    ) -> Sequence[RetrievedChunk]:
        """Return ranked evidence chunks for a lexical and semantic query."""
        ...


class ChunkIdentityIndex(Protocol):
    """Port for checking whether annotated evidence IDs remain indexed."""

    def existing_chunk_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        """Return the requested IDs that exist in the evidence corpus."""
        ...


def retrieve_evidence_chunks(
    *,
    query: str,
    retriever: HybridChunkRetriever,
    embedding_service: QueryEmbedder,
    limit: int = 10,
    candidate_limit: int | None = None,
    filters: MetadataFilter | None = None,
) -> dict[str, str]:
    """Return ranked evidence chunk IDs mapped to their speech text."""
    results = retrieve_ranked_evidence(
        query=query,
        retriever=retriever,
        embedding_service=embedding_service,
        limit=limit,
        candidate_limit=candidate_limit,
        filters=filters,
    )
    return {result.chunk_id: result.chunk_text for result in results}


def retrieve_ranked_evidence(
    *,
    query: str,
    retriever: HybridChunkRetriever,
    embedding_service: QueryEmbedder,
    limit: int = 10,
    candidate_limit: int | None = None,
    filters: MetadataFilter | None = None,
) -> Sequence[RetrievedChunk]:
    """Return ranked evidence chunks with citation provenance intact."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    if candidate_limit is not None and candidate_limit <= 0:
        raise ValueError("candidate_limit must be greater than zero")

    query_embedding = embed_query(query, embedding_service=embedding_service)
    return retriever.search_hybrid(
        query=query,
        query_embedding=query_embedding,
        limit=limit,
        candidate_limit=candidate_limit,
        filters=filters,
    )


def embed_query(
    query: str,
    *,
    embedding_service: QueryEmbedder,
) -> list[float]:
    """Validate a question and return its numeric query embedding."""
    if not query.strip():
        raise ValueError("query must not be empty")

    embedded_query = embedding_service.embed_one(query, prompt_name="query")
    if isinstance(embedded_query, Tensor):
        embedded_query = embedded_query.tolist()
    if not isinstance(embedded_query, list) or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in embedded_query
    ):
        raise TypeError("embedding_service must return a one-dimensional numeric query embedding")
    query_embedding = [float(value) for value in embedded_query]
    if not query_embedding:
        raise ValueError("embedding_service returned an empty query embedding")

    return query_embedding
