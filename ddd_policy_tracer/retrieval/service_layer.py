"""Graph ingestion and chunk-embedding retrieval services."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any, Protocol, Sequence

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.discovery.domain import SourceDocument


@dataclass(frozen=True)
class ChunkSimilarity:
    """Capture one chunk similarity score for a retrieval query."""

    chunk_id: str
    source_id: str
    source_document_id: str
    chunk_text: str
    similarity: float


class GraphStore(Protocol):
    """Minimal graph contract for ingestion and similarity search."""

    def upsert_source_document(self, document: SourceDocument) -> None:
        """Persist or update one source document node."""

    def upsert_chunk(self, chunk: DocumentChunk) -> None:
        """Persist or update one chunk node and its relationships."""

    def query_similar_chunks(
        self,
        *,
        embedding: Sequence[float],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the top similar chunks for a given embedding."""


class GraphIngestionService:
    """Coordinate the write-path for source documents and chunk nodes."""

    def __init__(self, *, graph_store: GraphStore) -> None:
        """Bind the ingestion service to a graph store implementation."""
        self._graph_store = graph_store

    def ingest_source_document(self, document: SourceDocument) -> None:
        """Persist a source document graph node."""
        self._graph_store.upsert_source_document(document)

    def ingest_document_chunk(self, chunk: DocumentChunk) -> None:
        """Persist a chunk node and attach it to the owning document."""
        self._graph_store.upsert_chunk(chunk)

    def ingest_document(
        self,
        *,
        document: SourceDocument,
        chunks: list[DocumentChunk],
    ) -> None:
        """Persist a source document and all of its chunks in one call."""
        self.ingest_source_document(document)
        for chunk in chunks:
            self.ingest_document_chunk(chunk)


class ChunkEmbeddingRetriever:
    """Query the graph using chunk embedding similarity."""

    def __init__(self, *, graph_store: GraphStore) -> None:
        """Bind the retriever to a graph implementation."""
        self._graph_store = graph_store

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        limit: int = 10,
    ) -> list[ChunkSimilarity]:
        """Return the best chunk matches by cosine similarity."""
        rows = self._graph_store.query_similar_chunks(
            embedding=query_embedding,
            limit=limit,
        )
        return [
            ChunkSimilarity(
                chunk_id=str(row["chunk_id"]),
                source_id=str(row["source_id"]),
                source_document_id=str(row["source_document_id"]),
                chunk_text=str(row["chunk_text"]),
                similarity=float(row["similarity"]),
            )
            for row in rows
        ]


class FalkorDBGraphStore:
    """Persist source documents and chunk embeddings in a FalkorDB graph."""

    def __init__(
        self,
        *,
        graph_name: str = "policy_tracer",
        host: str = "localhost",
        port: int = 6379,
        client: Any | None = None,
    ) -> None:
        """Create a client-backed graph store for FalkorDB."""
        if client is None:
            try:
                from falkordb import FalkorDB
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise RuntimeError(
                    "falkordb is required to use FalkorDBGraphStore. "
                    "Install the falkordb package or pass a custom client.",
                ) from exc
            client = FalkorDB(host=host, port=port)

        self._graph = _resolve_graph(client=client, graph_name=graph_name)

    def upsert_source_document(self, document: SourceDocument) -> None:
        """Merge a source document node keyed by source_document_id."""
        self._graph.query(
            """
            MERGE (document:SourceDocument {source_document_id: $source_document_id})
            SET document.source_id = $source_id,
                document.source_url = $source_url,
                document.published_at = $published_at,
                document.retrieved_at = $retrieved_at,
                document.checksum = $checksum,
                document.normalized_text = $normalized_text,
                document.raw_content_ref = $raw_content_ref,
                document.content_type = $content_type,
                document.created_at = $created_at,
                document.updated_at = $updated_at
            """,
            {
                "source_document_id": document.source_document_id,
                "source_id": document.source_id,
                "source_url": document.source_url,
                "published_at": document.published_at,
                "retrieved_at": document.retrieved_at,
                "checksum": document.checksum,
                "normalized_text": document.normalized_text,
                "raw_content_ref": document.raw_content_ref,
                "content_type": document.content_type,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
            },
        )

    def upsert_chunk(self, chunk: DocumentChunk) -> None:
        """Merge a chunk node and connect it to its source document."""
        self._graph.query(
            """
            MATCH (document:SourceDocument {source_document_id: $source_document_id})
            MERGE (chunk:DocumentChunk {chunk_id: $chunk_id})
            SET chunk.source_id = $source_id,
                chunk.source_document_id = $source_document_id,
                chunk.document_checksum = $document_checksum,
                chunk.chunk_index = $chunk_index,
                chunk.start_char = $start_char,
                chunk.end_char = $end_char,
                chunk.chunk_text = $chunk_text,
                chunk.embedding = $embedding
            MERGE (document)-[:HAS_CHUNK]->(chunk)
            """,
            {
                "source_document_id": chunk.source_document_id,
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "document_checksum": chunk.document_checksum,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "chunk_text": chunk.chunk_text,
                "embedding": list(chunk.embeddings or []),
            },
        )

    def query_similar_chunks(
        self,
        *,
        embedding: Sequence[float],
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Load all stored chunk embeddings and rank them by cosine similarity."""
        candidates = self._graph.query(
            """
            MATCH (chunk:DocumentChunk)
            WHERE chunk.embedding IS NOT NULL
            RETURN chunk.chunk_id AS chunk_id,
                   chunk.source_id AS source_id,
                   chunk.source_document_id AS source_document_id,
                   chunk.chunk_text AS chunk_text,
                   chunk.embedding AS embedding
            """,
        )

        matches: list[dict[str, Any]] = []
        target = list(embedding)
        for row in _iter_query_rows(candidates):
            if "embedding" not in row or row["embedding"] is None:
                continue
            candidate_vector = list(row["embedding"])
            similarity = _cosine_similarity(target, candidate_vector)
            matches.append(
                {
                    "chunk_id": row["chunk_id"],
                    "source_id": row["source_id"],
                    "source_document_id": row["source_document_id"],
                    "chunk_text": row["chunk_text"],
                    "similarity": similarity,
                }
            )
        matches.sort(key=lambda item: float(item["similarity"]), reverse=True)
        return matches[:limit]


def _resolve_graph(*, client: Any, graph_name: str) -> Any:
    """Resolve a graph handle from a FalkorDB client object."""
    if hasattr(client, "select_graph"):
        return client.select_graph(graph_name)
    if hasattr(client, "graph"):
        return client.graph(graph_name)
    raise TypeError("Unsupported FalkorDB client object: expected a select_graph or graph method.")


def _iter_query_rows(rows: Any) -> list[dict[str, Any]]:
    """Normalise graph query rows from various client return shapes."""
    if rows is None:
        return []
    if isinstance(rows, list):
        normalized: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                normalized.append(row)
            elif isinstance(row, tuple):
                normalized.append({f"col_{index}": value for index, value in enumerate(row)})
        return normalized
    if isinstance(rows, tuple):
        return [{"col_0": rows}]
    return [dict(rows)]


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity for two equal-length numeric vectors."""
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0
    dot_product = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = sqrt(sum(a * a for a in left_values))
    right_norm = sqrt(sum(a * a for a in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)
