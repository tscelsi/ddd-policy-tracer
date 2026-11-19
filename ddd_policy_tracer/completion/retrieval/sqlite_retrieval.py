"""SQLite-backed vector, BM25, and hybrid retrieval for document chunks."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock

import sqlite_vec  # type: ignore[import-untyped]
from sqlite_vec import serialize_float32  # type: ignore[import-untyped]

from ddd_policy_tracer.completion.filtering.metadata_filters import (
    MetadataFilter,
)
from ddd_policy_tracer.utils.time_helpers import utc_published_at_bounds

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
    published_at: str | None = None
    speaker_name: str | None = None
    talker_id: str | None = None
    party: str | None = None
    electorate: str | None = None
    house: str | None = None
    chamber: str | None = None
    bill_ids: list[str] | None = None
    debate_category: str | None = None
    debate_titles: list[str] | None = None
    speech_part_type: str | None = None


class SQLiteHybridRetriever:
    """Search SQLite chunk text and sqlite-vec embeddings with hybrid ranking."""

    def __init__(self, database_path: Path) -> None:
        """Build an in-memory FTS5 index from the read-only chunk database."""
        resolved_path = database_path.resolve(strict=True)
        if not resolved_path.is_file():
            raise FileNotFoundError(f"SQLite retrieval database is not a file: {resolved_path}")
        self._database_uri = f"{resolved_path.as_uri()}?mode=ro"
        self._fts_lock = Lock()
        self._fts_connection = sqlite3.connect(":memory:", check_same_thread=False)
        self._create_fts_index()

    def __enter__(self) -> SQLiteHybridRetriever:
        """Provide managed lifetime for the in-memory lexical index."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release the in-memory lexical index."""
        self.close()

    def close(self) -> None:
        """Close the in-memory lexical index after retrieval is complete."""
        with self._fts_lock:
            self._fts_connection.close()

    def existing_chunk_ids(self, chunk_ids: Sequence[str]) -> set[str]:
        """Return annotated chunk IDs that still exist in the indexed corpus."""
        if not chunk_ids:
            return set()
        connection = self._open_database(load_vector_extension=False)
        try:
            rows = connection.execute(
                """
                SELECT chunk_id
                FROM document_chunks
                WHERE chunk_id IN (SELECT value FROM json_each(?))
                """,
                (json.dumps(list(chunk_ids)),),
            ).fetchall()
        finally:
            connection.close()
        return {str(row[0]) for row in rows}

    def search_vector(
        self,
        *,
        query_embedding: Sequence[float],
        limit: int = 10,
        filters: MetadataFilter | None = None,
    ) -> list[SQLiteChunkResult]:
        """Return nearest eligible chunks using the database's sqlite-vec index."""
        _validate_limit(limit)
        if not query_embedding:
            raise ValueError("query_embedding must contain at least one value")
        query_vector = serialize_float32([float(value) for value in query_embedding])
        filter_parameters = _metadata_filter_parameters(filters)
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
                       chunks.published_at,
                       chunks.speaker_name,
                       chunks.talker_id,
                       chunks.party,
                       chunks.electorate,
                       chunks.house,
                       chunks.chamber,
                       chunks.bill_ids,
                       chunks.debate_category,
                       chunks.debate_titles,
                       chunks.speech_part_type,
                       nearest.distance
                FROM (
                    SELECT chunk_id, distance
                    FROM chunk_embeddings
                    WHERE embedding MATCH ? AND k = ?
                      AND (
                        ? = 1 OR chunk_id IN (
                            SELECT chunk_id
                            FROM document_chunks
                            WHERE (? = 1 OR talker_id IN (
                                SELECT value FROM json_each(?)
                            ))
                              AND (? = 1 OR (
                                source_id = 'hansard'
                                AND (? = 1 OR published_at >= ?)
                                AND (? = 1 OR published_at < ?)
                            ))
                        )
                      )
                    ORDER BY distance
                ) AS nearest
                JOIN document_chunks AS chunks
                  ON chunks.chunk_id = nearest.chunk_id
                ORDER BY nearest.distance
                """,
                (query_vector, limit, *filter_parameters),
            ).fetchall()
        finally:
            connection.close()

        return [_chunk_result(row, vector_distance=float(row[20])) for row in rows]

    def search_bm25(
        self,
        *,
        query: str,
        limit: int = 10,
        filters: MetadataFilter | None = None,
    ) -> list[SQLiteChunkResult]:
        """Return eligible chunks ranked by SQLite FTS5 BM25 over their text."""
        _validate_limit(limit)
        fts_query = _to_fts_query(query)
        if not fts_query:
            return []
        filter_parameters = _metadata_filter_parameters(filters)

        with self._fts_lock:
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
                       published_at,
                       speaker_name,
                       talker_id,
                       party,
                       electorate,
                       house,
                       chamber,
                       bill_ids,
                       debate_category,
                       debate_titles,
                       speech_part_type,
                       bm25(chunks_fts, 0, 0, 0, 0, 0, 0, 0, 0, 1.0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                  AND (
                    ? = 1 OR (
                        (? = 1 OR talker_id IN (
                            SELECT value FROM json_each(?)
                        ))
                        AND (? = 1 OR (
                            source_id = 'hansard'
                            AND (? = 1 OR published_at >= ?)
                            AND (? = 1 OR published_at < ?)
                        ))
                    )
                  )
                ORDER BY score ASC
                LIMIT ?
                """,
                (fts_query, *filter_parameters, limit),
            ).fetchall()
        return [_chunk_result(row, bm25_score=float(row[20])) for row in rows]

    def search_hybrid(
        self,
        *,
        query: str,
        query_embedding: Sequence[float],
        limit: int = 10,
        candidate_limit: int | None = None,
        filters: MetadataFilter | None = None,
    ) -> list[SQLiteChunkResult]:
        """Filter vector and BM25 candidates before reciprocal-rank fusion."""
        _validate_limit(limit)
        candidates = candidate_limit if candidate_limit is not None else max(30, limit * 3)
        _validate_limit(candidates, name="candidate_limit")

        vector_results = self.search_vector(
            query_embedding=query_embedding,
            limit=candidates,
            filters=filters,
        )
        lexical_results = self.search_bm25(
            query=query,
            limit=candidates,
            filters=filters,
        )
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
                published_at UNINDEXED,
                speaker_name UNINDEXED,
                talker_id UNINDEXED,
                party UNINDEXED,
                electorate UNINDEXED,
                house UNINDEXED,
                chamber UNINDEXED,
                bill_ids UNINDEXED,
                debate_category UNINDEXED,
                debate_titles UNINDEXED,
                speech_part_type UNINDEXED,
                tokenize='unicode61'
            )
            """,
        )
        connection = self._open_database(load_vector_extension=False)
        try:
            rows = connection.execute(
                """
                SELECT chunk_id, source_id, source_document_id, chunk_index,
                       start_char, end_char, topics, metadata, chunk_text,
                       published_at, speaker_name, talker_id, party, electorate,
                       house, chamber, bill_ids, debate_category, debate_titles,
                       speech_part_type
                FROM document_chunks
                """,
            )
            self._fts_connection.executemany(
                """
                INSERT INTO chunks_fts (
                    chunk_id, source_id, source_document_id, chunk_index,
                    start_char, end_char, topics, metadata, chunk_text,
                    published_at, speaker_name, talker_id, party, electorate,
                    house, chamber, bill_ids, debate_category, debate_titles,
                    speech_part_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        *row[:9],
                        *row[9:],
                    )
                    for row in rows
                ),
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
        published_at=_optional_str(row[9]),
        speaker_name=_optional_str(row[10]),
        talker_id=_optional_str(row[11]),
        party=_optional_str(row[12]),
        electorate=_optional_str(row[13]),
        house=_optional_str(row[14]),
        chamber=_optional_str(row[15]),
        bill_ids=_json_string_list(row[16]),
        debate_category=_optional_str(row[17]),
        debate_titles=_json_string_list(row[18]),
        speech_part_type=_optional_str(row[19]),
        vector_distance=vector_distance,
        bm25_score=bm25_score,
    )


def _optional_str(value: object) -> str | None:
    """Return a retrieved field only when SQLite returned text."""
    return value if isinstance(value, str) else None


def _json_string_list(value: object) -> list[str] | None:
    """Decode a JSON array of metadata strings from SQLite."""
    if not isinstance(value, str):
        return None
    decoded = json.loads(value)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise ValueError("Expected a JSON array of strings in chunk metadata.")
    return decoded


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


def _metadata_filter_parameters(
    filters: MetadataFilter | None,
) -> tuple[object, ...]:
    """Return parameterized constraints for the fixed vector and FTS SQL statements."""
    has_filter = filters is not None and not filters.is_empty
    has_speakers = filters is not None and bool(filters.talker_ids)
    has_dates = filters is not None and (
        filters.date_from is not None or filters.date_to is not None
    )
    start, end = utc_published_at_bounds(
        date_from=filters.date_from if filters is not None else None,
        date_to=filters.date_to if filters is not None else None,
    )
    talker_ids = json.dumps(filters.talker_ids if filters is not None else ())
    return (
        int(not has_filter),
        int(not has_speakers),
        talker_ids,
        int(not has_dates),
        int(start is None),
        start,
        int(end is None),
        end,
    )
