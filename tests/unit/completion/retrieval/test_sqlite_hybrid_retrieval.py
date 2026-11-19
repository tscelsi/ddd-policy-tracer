"""Tests for SQLite vector and BM25 retrieval."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Literal

import pytest
import sqlite_vec  # type: ignore[import-untyped]
from sqlite_vec import serialize_float32
from torch import Tensor

from ddd_policy_tracer.completion.filtering.metadata_filters import MetadataFilter
from ddd_policy_tracer.completion.retrieval.service_layer import retrieve_evidence_chunks
from ddd_policy_tracer.completion.retrieval.sqlite_retrieval import SQLiteHybridRetriever


class StubEmbeddingService:
    """Return a fixed query vector while recording the embedded query."""

    def __init__(self) -> None:
        """Prepare query state for retrieval service tests."""
        self.query: str | None = None
        self.prompt_name: Literal["query", "document"] | None = None

    def embed_one(
        self,
        text: str,
        prompt_name: Literal["query", "document"] = "document",
        return_tensor: bool = False,
    ) -> list[float] | Tensor:
        """Record the question and return the vector matching the test corpus."""
        self.query = text
        self.prompt_name = prompt_name
        return [0.0, 1.0, 0.0]


def _create_database(path: Path) -> None:
    """Create a small database with chunk text and sqlite-vec embeddings."""
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE document_chunks (
            chunk_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            source_document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            topics TEXT,
            metadata TEXT,
            published_at TEXT,
            speaker_name TEXT,
            talker_id TEXT,
            party TEXT,
            electorate TEXT,
            house TEXT,
            chamber TEXT,
            bill_ids TEXT,
            debate_category TEXT,
            debate_titles TEXT,
            speech_part_type TEXT
        )
        """,
    )
    connection.enable_load_extension(True)
    try:
        sqlite_vec.load(connection)
    finally:
        connection.enable_load_extension(False)
    connection.execute(
        """
        CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
            chunk_id TEXT PRIMARY KEY,
            embedding float[3]
        )
        """,
    )
    chunks = [
        (
            "chunk-vector",
            "hansard",
            "speech-1",
            0,
            0,
            40,
            "Climate policy and renewable energy.",
            '["Environment"]',
            '{"speaker":"Member A"}',
            [0.0, 1.0, 0.0],
        ),
        (
            "chunk-lexical",
            "hansard",
            "speech-2",
            0,
            0,
            42,
            "Renewable energy supports regional jobs.",
            '["Energy"]',
            '{"speaker":"Member B"}',
            [1.0, 0.0, 0.0],
        ),
        (
            "chunk-unrelated",
            "hansard",
            "speech-3",
            0,
            0,
            26,
            "The budget funds local roads.",
            None,
            None,
            [0.0, 0.0, 1.0],
        ),
    ]
    connection.executemany(
        """
        INSERT INTO document_chunks (
            chunk_id, source_id, source_document_id, chunk_index,
            start_char, end_char, chunk_text, topics, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [chunk[:9] for chunk in chunks],
    )
    connection.executemany(
        "INSERT INTO chunk_embeddings (chunk_id, embedding) VALUES (?, ?)",
        [(chunk[0], serialize_float32(chunk[9])) for chunk in chunks],
    )
    connection.execute(
        """
        UPDATE document_chunks
        SET published_at = '2024-06-01',
            speaker_name = 'Member A',
            talker_id = 'talker-1',
            party = 'Example Party',
            electorate = 'Example Electorate',
            house = 'senate',
            chamber = 'main',
            bill_ids = '["bill-1"]',
            debate_category = 'Bills',
            debate_titles = '["Second reading"]',
            speech_part_type = 'speech'
        WHERE chunk_id = 'chunk-vector'
        """,
    )
    connection.execute(
        """
        UPDATE document_chunks
        SET published_at = '2024-06-02',
            speaker_name = 'Member B',
            talker_id = 'talker-2',
            house = 'senate'
        WHERE chunk_id = 'chunk-lexical'
        """,
    )
    connection.execute(
        """
        UPDATE document_chunks
        SET source_id = 'other',
            published_at = '2024-06-01',
            talker_id = 'talker-1'
        WHERE chunk_id = 'chunk-unrelated'
        """,
    )
    connection.commit()
    connection.close()


def test_vector_search_returns_nearest_chunk_and_provenance(tmp_path: Path) -> None:
    """Find the closest stored speech chunk and preserve its metadata."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        results = retriever.search_vector(query_embedding=[0.0, 1.0, 0.0], limit=2)
        lexical_results = retriever.search_bm25(query="climate", limit=1)

    assert results[0].chunk_id == "chunk-vector"
    assert results[0].vector_distance == 0.0
    assert results[0].source_document_id == "speech-1"
    assert results[0].topics == '["Environment"]'
    assert results[0].metadata == '{"speaker":"Member A"}'
    assert results[0].published_at == "2024-06-01"
    assert results[0].speaker_name == "Member A"
    assert results[0].talker_id == "talker-1"
    assert results[0].party == "Example Party"
    assert results[0].electorate == "Example Electorate"
    assert results[0].house == "senate"
    assert results[0].chamber == "main"
    assert results[0].bill_ids == ["bill-1"]
    assert results[0].debate_category == "Bills"
    assert results[0].debate_titles == ["Second reading"]
    assert results[0].speech_part_type == "speech"
    assert results[0].bm25_score is None
    assert lexical_results[0].chunk_id == "chunk-vector"
    assert lexical_results[0].talker_id == "talker-1"


def test_existing_chunk_ids_reports_only_ids_in_the_corpus(tmp_path: Path) -> None:
    """Distinguish stale evidence annotations from retrieval ranking misses."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        existing_ids = retriever.existing_chunk_ids(["chunk-vector", "deleted-chunk"])

    assert existing_ids == {"chunk-vector"}


def test_bm25_search_ranks_matching_chunk_and_quotes_query_terms(tmp_path: Path) -> None:
    """Rank exact lexical matches without treating user text as FTS syntax."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        results = retriever.search_bm25(query='renewable "regional" OR *', limit=3)

    assert results
    assert results[0].chunk_id == "chunk-lexical"
    assert results[0].bm25_score is not None
    assert results[0].vector_distance is None


def test_hybrid_search_fuses_vector_and_bm25_rankings(tmp_path: Path) -> None:
    """Merge semantic and lexical candidates with reciprocal-rank fusion."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        results = retriever.search_hybrid(
            query="renewable energy",
            query_embedding=[0.0, 1.0, 0.0],
            limit=2,
        )

    by_id = {result.chunk_id: result for result in results}
    assert set(by_id) == {"chunk-vector", "chunk-lexical"}
    assert all(result.fused_score is not None for result in results)
    assert by_id["chunk-vector"].vector_distance == 0.0
    assert by_id["chunk-lexical"].bm25_score is not None


def test_hybrid_search_applies_filters_before_candidate_limits(tmp_path: Path) -> None:
    """Keep matching chunks eligible for both ranks before each candidate top-k."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        results = retriever.search_hybrid(
            query="renewable energy",
            query_embedding=[1.0, 0.0, 0.0],
            limit=3,
            candidate_limit=1,
            filters=MetadataFilter(
                talker_ids=("talker-1",),
                date_from=date(2024, 6, 1),
                date_to=date(2024, 6, 1),
            ),
        )

    assert [result.chunk_id for result in results] == ["chunk-vector"]
    assert results[0].talker_id == "talker-1"


def test_hybrid_search_combines_speaker_or_and_date_and(tmp_path: Path) -> None:
    """Match any requested speaker while requiring all selected filter fields."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        results = retriever.search_hybrid(
            query="renewable energy",
            query_embedding=[0.0, 1.0, 0.0],
            limit=3,
            filters=MetadataFilter(
                talker_ids=("talker-1", "talker-2"),
                date_from=date(2024, 6, 2),
                date_to=date(2024, 6, 2),
            ),
        )

    assert results
    assert {result.chunk_id for result in results} == {"chunk-lexical"}


def test_hybrid_search_does_not_fall_back_when_filters_match_nothing(tmp_path: Path) -> None:
    """Preserve empty results instead of retrying an unfiltered query."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        results = retriever.search_hybrid(
            query="renewable energy",
            query_embedding=[0.0, 1.0, 0.0],
            filters=MetadataFilter(talker_ids=("missing-talker",)),
        )

    assert results == []


def test_hybrid_search_supports_a_worker_thread(tmp_path: Path) -> None:
    """Search with the startup-created FTS index from an API worker thread."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    retriever.search_hybrid,
                    query="renewable energy",
                    query_embedding=[0.0, 1.0, 0.0],
                    limit=2,
                )
                for _ in range(4)
            ]
            results = [future.result() for future in futures]

    assert all(
        {result.chunk_id for result in search_results} == {"chunk-vector", "chunk-lexical"}
        for search_results in results
    )


def test_bm25_search_returns_no_results_for_nonmatching_query(tmp_path: Path) -> None:
    """Return no candidates when the query has no indexed terms."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        assert retriever.search_bm25(query="   !!! ", limit=3) == []


def test_search_rejects_nonpositive_limit(tmp_path: Path) -> None:
    """Reject result limits that cannot produce a useful ranked list."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        with pytest.raises(ValueError, match="limit"):
            retriever.search_bm25(query="energy", limit=0)


def test_retrieve_evidence_chunks_returns_ranked_ids_and_text(tmp_path: Path) -> None:
    """Embed a question and return hybrid-ranked chunk IDs with their text."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)
    embedding_service = StubEmbeddingService()

    with SQLiteHybridRetriever(database_path) as retriever:
        results = retrieve_evidence_chunks(
            query="renewable energy",
            retriever=retriever,
            embedding_service=embedding_service,
            limit=2,
        )

    assert results == {
        "chunk-vector": "Climate policy and renewable energy.",
        "chunk-lexical": "Renewable energy supports regional jobs.",
    }
    assert embedding_service.query == "renewable energy"
    assert embedding_service.prompt_name == "query"


def test_retrieve_evidence_chunks_rejects_empty_query(tmp_path: Path) -> None:
    """Reject an empty question before calling the embedding service."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)
    embedding_service = StubEmbeddingService()

    with SQLiteHybridRetriever(database_path) as retriever:
        with pytest.raises(ValueError, match="query"):
            retrieve_evidence_chunks(
                query="  ",
                retriever=retriever,
                embedding_service=embedding_service,
            )

    assert embedding_service.query is None
