"""Tests for SQLite vector and BM25 retrieval."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import sqlite_vec  # type: ignore[import-untyped]
from sqlite_vec import serialize_float32

from ddd_policy_tracer.retrieval.sqlite_retrieval import SQLiteHybridRetriever


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
            metadata TEXT
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
    connection.commit()
    connection.close()


def test_vector_search_returns_nearest_chunk_and_provenance(tmp_path: Path) -> None:
    """Find the closest stored speech chunk and preserve its metadata."""
    database_path = tmp_path / "retrieval.db"
    _create_database(database_path)

    with SQLiteHybridRetriever(database_path) as retriever:
        results = retriever.search_vector(query_embedding=[0.0, 1.0, 0.0], limit=2)

    assert results[0].chunk_id == "chunk-vector"
    assert results[0].vector_distance == 0.0
    assert results[0].source_document_id == "speech-1"
    assert results[0].topics == '["Environment"]'
    assert results[0].metadata == '{"speaker":"Member A"}'
    assert results[0].bm25_score is None


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
