from __future__ import annotations

from ddd_policy_tracer.analysis.chunks.chunking_models import DocumentChunk
from ddd_policy_tracer.discovery.domain import SourceDocument
from ddd_policy_tracer.retrieval.service_layer import ChunkEmbeddingRetriever, GraphIngestionService


class InMemoryGraphStore:
    def __init__(self) -> None:
        self.source_documents: dict[str, dict[str, object]] = {}
        self.chunks: dict[str, dict[str, object]] = {}

    def upsert_source_document(self, document: SourceDocument) -> None:
        self.source_documents[document.source_document_id] = {
            "source_id": document.source_id,
            "source_document_id": document.source_document_id,
            "source_url": document.source_url,
            "retrieved_at": document.retrieved_at,
            "normalized_text": document.normalized_text,
            "checksum": document.checksum,
        }

    def upsert_chunk(self, chunk: DocumentChunk) -> None:
        self.chunks[chunk.chunk_id] = {
            "chunk_id": chunk.chunk_id,
            "source_id": chunk.source_id,
            "source_document_id": chunk.source_document_id,
            "chunk_text": chunk.chunk_text,
            "embeddings": list(chunk.embeddings or []),
        }

    def query_similar_chunks(
        self,
        *,
        embedding: list[float],
        limit: int = 10,
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        for chunk_data in self.chunks.values():
            candidate = list(chunk_data["embeddings"])
            if not candidate:
                continue
            similarity = _cosine_similarity(embedding, candidate)
            matches.append(
                {
                    "chunk_id": chunk_data["chunk_id"],
                    "source_id": chunk_data["source_id"],
                    "source_document_id": chunk_data["source_document_id"],
                    "chunk_text": chunk_data["chunk_text"],
                    "similarity": similarity,
                }
            )
        return sorted(matches, key=lambda row: row["similarity"], reverse=True)[:limit]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(a * a for a in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def test_graph_ingestion_creates_document_chunk_relationship() -> None:
    store = InMemoryGraphStore()
    service = GraphIngestionService(graph_store=store)
    document = SourceDocument(
        source_id="source-1",
        source_document_id="doc-1",
        source_url="https://example.com/doc-1",
        published_at="2024-01-01",
        retrieved_at="2024-01-02T00:00:00Z",
        checksum="abc123",
        normalized_text="Policy for renewable energy.",
        raw_content_ref="raw://doc-1",
        content_type="text/html",
        created_at="2024-01-02T00:00:00Z",
        updated_at="2024-01-02T00:00:00Z",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        source_id="source-1",
        source_document_id="doc-1",
        document_checksum="abc123",
        chunk_index=0,
        start_char=0,
        end_char=10,
        chunk_text="Policy for renewable energy.",
        embeddings=[1.0, 0.0, 0.0],
    )

    service.ingest_source_document(document)
    service.ingest_document_chunk(chunk)

    assert store.source_documents["doc-1"]["source_document_id"] == "doc-1"
    assert store.chunks["chunk-1"]["source_document_id"] == "doc-1"


def test_chunk_embedding_search_ranks_similar_chunks() -> None:
    store = InMemoryGraphStore()
    service = GraphIngestionService(graph_store=store)
    service.ingest_source_document(
        SourceDocument(
            source_id="source-1",
            source_document_id="doc-1",
            source_url="https://example.com/doc-1",
            published_at="2024-01-01",
            retrieved_at="2024-01-02T00:00:00Z",
            checksum="one",
            normalized_text="Climate policy.",
            raw_content_ref="raw://1",
            content_type="text/html",
            created_at="2024-01-02T00:00:00Z",
            updated_at="2024-01-02T00:00:00Z",
        )
    )
    service.ingest_document_chunk(
        DocumentChunk(
            chunk_id="chunk-1",
            source_id="source-1",
            source_document_id="doc-1",
            document_checksum="one",
            chunk_index=0,
            start_char=0,
            end_char=7,
            chunk_text="Climate policy.",
            embeddings=[1.0, 0.0, 0.0],
        )
    )
    service.ingest_document_chunk(
        DocumentChunk(
            chunk_id="chunk-2",
            source_id="source-1",
            source_document_id="doc-1",
            document_checksum="one",
            chunk_index=1,
            start_char=8,
            end_char=20,
            chunk_text="Energy security.",
            embeddings=[0.0, 1.0, 0.0],
        )
    )

    retriever = ChunkEmbeddingRetriever(graph_store=store)
    matches = retriever.search(query_embedding=[1.0, 0.0, 0.0], limit=2)

    assert matches[0].chunk_id == "chunk-1"
    assert matches[0].similarity > 0.99
