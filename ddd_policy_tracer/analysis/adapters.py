"""Persistence adapters for document chunk storage backends."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .chunking_models import DocumentChunk


class SQLiteDocumentChunkRepository:
    """Store and retrieve document chunks from SQLite."""

    def __init__(self, sqlite_path: Path) -> None:
        """Bind chunk repository to a SQLite state path."""
        self._sqlite_path = sqlite_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection for chunk repository operations."""
        return sqlite3.connect(self._sqlite_path)

    def _initialize(self) -> None:
        """Create chunk table and uniqueness constraints when needed."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    document_checksum TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    UNIQUE (
                        source_id,
                        source_document_id,
                        document_checksum,
                        chunk_index
                    )
                )
                """,
            )

    def has_chunks_for_document_version(
        self,
        *,
        source_id: str,
        source_document_id: str,
        document_checksum: str,
    ) -> bool:
        """Return true when chunks already exist for one document version."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM document_chunks
                WHERE source_id = ?
                  AND source_document_id = ?
                  AND document_checksum = ?
                LIMIT 1
                """,
                (source_id, source_document_id, document_checksum),
            ).fetchone()
        return row is not None

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Persist chunks and return number of inserted records."""
        if not chunks:
            return 0

        with self._connect() as connection:
            cursor = connection.executemany(
                """
                INSERT OR IGNORE INTO document_chunks (
                    chunk_id,
                    source_id,
                    source_document_id,
                    document_checksum,
                    chunk_index,
                    start_char,
                    end_char,
                    chunk_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.source_id,
                        chunk.source_document_id,
                        chunk.document_checksum,
                        chunk.chunk_index,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.chunk_text,
                    )
                    for chunk in chunks
                ],
            )
        return cursor.rowcount

    def list_chunks(
        self,
        *,
        source_id: str,
        source_document_id: str,
        document_checksum: str,
    ) -> list[DocumentChunk]:
        """List all chunks for one document version ordered by index."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, source_id, source_document_id,
                       document_checksum, chunk_index,
                       start_char, end_char, chunk_text
                FROM document_chunks
                WHERE source_id = ?
                  AND source_document_id = ?
                  AND document_checksum = ?
                ORDER BY chunk_index ASC
                """,
                (source_id, source_document_id, document_checksum),
            ).fetchall()

        return [
            DocumentChunk(
                chunk_id=row[0],
                source_id=row[1],
                source_document_id=row[2],
                document_checksum=row[3],
                chunk_index=row[4],
                start_char=row[5],
                end_char=row[6],
                chunk_text=row[7],
            )
            for row in rows
        ]


class FilesystemDocumentChunkRepository:
    """Store and retrieve document chunks from JSONL state."""

    def __init__(self, state_path: Path) -> None:
        """Bind chunk repository to one JSONL state file path."""
        self._state_path = state_path
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

    def has_chunks_for_document_version(
        self,
        *,
        source_id: str,
        source_document_id: str,
        document_checksum: str,
    ) -> bool:
        """Return true when JSONL already has chunks for a version."""
        for chunk in self._read_all():
            if (
                chunk.source_id == source_id
                and chunk.source_document_id == source_document_id
                and chunk.document_checksum == document_checksum
            ):
                return True
        return False

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Append chunk records to JSONL and return inserted count."""
        if not chunks:
            return 0

        with self._state_path.open("a", encoding="utf-8") as handle:
            for chunk in chunks:
                record = {
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "source_document_id": chunk.source_document_id,
                    "document_checksum": chunk.document_checksum,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "chunk_text": chunk.chunk_text,
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return len(chunks)

    def list_chunks(
        self,
        *,
        source_id: str,
        source_document_id: str,
        document_checksum: str,
    ) -> list[DocumentChunk]:
        """List JSONL chunks for one document version ordered by index."""
        matches = [
            chunk
            for chunk in self._read_all()
            if chunk.source_id == source_id
            and chunk.source_document_id == source_document_id
            and chunk.document_checksum == document_checksum
        ]
        return sorted(matches, key=lambda chunk: chunk.chunk_index)

    def _read_all(self) -> list[DocumentChunk]:
        """Read all chunk records from JSONL state."""
        if not self._state_path.exists():
            return []

        chunks: list[DocumentChunk] = []
        content = self._state_path.read_text(encoding="utf-8")
        for raw_line in content.splitlines():
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            chunks.append(
                DocumentChunk(
                    chunk_id=payload["chunk_id"],
                    source_id=payload["source_id"],
                    source_document_id=payload["source_document_id"],
                    document_checksum=payload["document_checksum"],
                    chunk_index=payload["chunk_index"],
                    start_char=payload["start_char"],
                    end_char=payload["end_char"],
                    chunk_text=payload["chunk_text"],
                ),
            )
        return chunks
