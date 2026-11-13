"""SQLite-backed vector, BM25, and hybrid retrieval for document chunks."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import sqlite_vec  # type: ignore[import-untyped]
from sqlite_vec import serialize_float32

_RRF_K = 60
_FTS_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class SQLiteChunkResult:
    """Represent one retrieved chunk with source provenance and retrieval scores."""

    chunk_id: str
    source_id: str
    source_document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    chunk_text: str
    topics: str | None
    metadata: str | None
    vector_distance: float | None = None
    bm25_score: float | None = None
    fused_score: float | None = None


class SQLiteHybridRetriever:
    """Search SQLite chunk text and sqlite-vec embeddings with hybrid ranking."""

    def __init__(self, database_path: Path) -> None:
        """Build an in-memory FTS5 index from the read-only chunk database."""
        resolved_path = database_path.resolve(strict=True)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"SQLite retrieval database is not a file: {resolved_path}")
        self._database_uri = f"{resolved_path.as_uri()}?mode=ro"
        self._fts_connection = sqlite3.connect(":memory:")
        self._create_fts_index()

    def __enter__(self) -> SQLiteHybridRetriever:
        """Provide managed lifetime for the in-memory lexical index."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the in-memory lexical index."""
        self.close()

    def close(self) -> None:
        """Close the in-memory lexical index after retrieval is complete."""
        self._fts_connection.close()

    def search_vector(
        self,
        *,
        query_embedding: Sequence[float],
        limit: int = 10,
    ) -> list[SQLiteChunkResult]:
        """Return nearest chunks using the database's sqlite-vec index."""
        _validate_limit(limit)
        if not query_embedding:
            raise ValueError("query_embedding must contain at least one value")
        query_vector = serialize_float32([float(value) for value in query_embedding])
        connection = self._open_database(load_vector_extension=True)
        try:
            rows = connection.execute(
                """
                SELECT chunks.chunk_id,
                       chunks.source_id,
                       chunks.source_document_id,
                       chunks.chunk_index,
                       chunks.start_char,
                       chunks.end_char,
                       chunks.chunk_text,
                       chunks.topics,
                       chunks.metadata,
                       nearest.distance
                FROM (
                    SELECT chunk_id, distance
                    FROM chunk_embeddings
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance
                ) AS nearest
                JOIN document_chunks AS chunks
                  ON chunks.chunk_id = nearest.chunk_id
                ORDER BY nearest.distance
                """,
                (query_vector, limit),
            ).fetchall()
        finally:
            connection.close()

        return [_chunk_result(row, vector_distance=float(row[9])) for row in rows]

    def search_bm25(self, *, query: str, limit: int = 10) -> list[SQLiteChunkResult]:
        """Return chunks ranked by SQLite FTS5 BM25 over their text."""
        _validate_limit(limit)
        fts_query = _to_fts_query(query)
        if not fts_query:
            return []

        rows = self._fts_connection.execute(
            """
            SELECT chunk_id,
                   source_id,
                   source_document_id,
                   chunk_index,
                   start_char,
                   end_char,
                   chunk_text,
                   topics,
                   metadata,
                   bm25(chunks_fts, 0, 0, 0, 0, 0, 0, 0, 0, 1.0) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return [_chunk_result(row, bm25_score=float(row[9])) for row in rows]

    def search_hybrid(
        self,
        *,
        query: str,
        query_embedding: Sequence[float],
        limit: int = 10,
        candidate_limit: int | None = None,
    ) -> list[SQLiteChunkResult]:
        """Combine vector and BM25 rankings using reciprocal-rank fusion."""
        _validate_limit(limit)
        candidates = candidate_limit if candidate_limit is not None else max(30, limit * 3)
        _validate_limit(candidates, name="candidate_limit")

        vector_results = self.search_vector(query_embedding=query_embedding, limit=candidates)
        lexical_results = self.search_bm25(query=query, limit=candidates)
        merged: dict[str, SQLiteChunkResult] = {}
        scores: dict[str, float] = {}

        for rank, result in enumerate(vector_results, start=1):
            merged[result.chunk_id] = result
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + _reciprocal_rank(rank)
        for rank, result in enumerate(lexical_results, start=1):
            existing = merged.get(result.chunk_id)
            if existing is None:
                merged[result.chunk_id] = result
            else:
                merged[result.chunk_id] = replace(existing, bm25_score=result.bm25_score)
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + _reciprocal_rank(rank)

        ranked = [
            replace(result, fused_score=scores[chunk_id]) for chunk_id, result in merged.items()
        ]
        return sorted(
            ranked,
            key=lambda result: result.fused_score or 0.0,
            reverse=True,
        )[:limit]

    def _create_fts_index(self) -> None:
        """Copy chunk text and provenance into an in-memory FTS5 index."""
        self._fts_connection.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                source_id UNINDEXED,
                source_document_id UNINDEXED,
                chunk_index UNINDEXED,
                start_char UNINDEXED,
                end_char UNINDEXED,
                topics UNINDEXED,
                metadata UNINDEXED,
                chunk_text,
                tokenize='unicode61'
            )
            """,
        )
        connection = self._open_database(load_vector_extension=False)
        try:
            rows = connection.execute(
                """
                SELECT chunk_id, source_id, source_document_id, chunk_index,
                       start_char, end_char, topics, metadata, chunk_text
                FROM document_chunks
                """,
            )
            self._fts_connection.executemany(
                """
                INSERT INTO chunks_fts (
                    chunk_id, source_id, source_document_id, chunk_index,
                    start_char, end_char, topics, metadata, chunk_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        finally:
            connection.close()

    def _open_database(self, *, load_vector_extension: bool) -> sqlite3.Connection:
        """Open the source database read-only and load sqlite-vec if needed."""
        connection = sqlite3.connect(self._database_uri, uri=True)
        if not load_vector_extension:
            return connection
        try:
            connection.enable_load_extension(True)
        except AttributeError as exc:
            connection.close()
            raise RuntimeError(
                "This Python SQLite build does not support loading sqlite-vec extensions.",
            ) from exc
        try:
            sqlite_vec.load(connection)
        finally:
            connection.enable_load_extension(False)
        return connection


def _chunk_result(
    row: tuple[object, ...],
    *,
    vector_distance: float | None = None,
    bm25_score: float | None = None,
) -> SQLiteChunkResult:
    """Map a database row into a provenance-bearing retrieval result."""
    return SQLiteChunkResult(
        chunk_id=str(row[0]),
        source_id=str(row[1]),
        source_document_id=str(row[2]),
        chunk_index=_as_int(row[3]),
        start_char=_as_int(row[4]),
        end_char=_as_int(row[5]),
        chunk_text=str(row[6]),
        topics=str(row[7]) if row[7] is not None else None,
        metadata=str(row[8]) if row[8] is not None else None,
        vector_distance=vector_distance,
        bm25_score=bm25_score,
    )


def _as_int(value: object) -> int:
    """Validate an integer value loaded from the SQLite chunk record."""
    if not isinstance(value, int):
        raise TypeError(f"Expected an integer chunk field, got {type(value).__name__}")
    return value


def _to_fts_query(query: str) -> str:
    """Convert natural-language input into safely quoted FTS5 terms."""
    terms = _FTS_TOKEN.findall(query)
    return " OR ".join(f'"{term}"' for term in terms)


def _reciprocal_rank(rank: int) -> float:
    """Give each ranked result its reciprocal-rank fusion contribution."""
    return 1.0 / (_RRF_K + rank)


def _validate_limit(limit: int, *, name: str = "limit") -> None:
    """Reject nonpositive retrieval result limits."""
    if limit <= 0:
        raise ValueError(f"{name} must be greater than zero")
