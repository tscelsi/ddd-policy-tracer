"""Persistence adapters for document chunk storage backends."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol

from ddd_policy_tracer.__DEPRECATED_analysis.entities._catalog.adapters import deserialize_f32, serialize_f32
from ddd_policy_tracer.utils.db_manager import DbManager

from .models import DocumentChunk

logger = logging.getLogger(__name__)


class AbstractChunkRepository(Protocol):
    """Protocol for document chunk repository implementations."""

    async def init(self) -> None:
        """Initialize the repository, creating tables or files as needed."""

    async def _ensure_schema(self) -> None:
        """Ensure the underlying storage schema is ready for operations."""

    async def save(self, chunk: DocumentChunk) -> None:
        """Persist one chunk, inserting or updating as needed."""

    async def save_bulk(self, *, chunks: list[DocumentChunk]) -> None:
        """Persist one or more document chunks."""

    async def save_embeddings(self, *, chunks: list[DocumentChunk]) -> None:
        """Persist one or more document embeddings."""

    async def get_chunk(self, *, chunk_id: str) -> DocumentChunk | None:
        """Return one chunk by identifier or none when missing."""

    async def list_chunks(
        self,
        *,
        chunk_ids: list[str] | None = None,
        source_id: str | None = None,
    ) -> list[DocumentChunk]:
        """List all chunks in the repository."""
        ...

    async def list_chunks_without_embedding(
        self,
        *,
        source_id: str | None = None,
    ) -> list[DocumentChunk]:
        """List all chunks without embeddings in the repository."""
        ...


class SQLiteDocumentChunkRepository(AbstractChunkRepository):
    """Store and retrieve document chunks from SQLite."""

    def __init__(self, db_manager: DbManager) -> None:
        """Bind chunk repository to one SQLite database manager."""
        self._db_manager = db_manager

    async def init(self) -> None:
        """Initialize the repository by ensuring the schema is ready."""
        await self._ensure_schema()

    async def _ensure_schema(self) -> None:
        async with self._db_manager.with_transaction() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY NOT NULL,
                    source_id TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    document_checksum TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    topics TEXT,
                    metadata TEXT,
                    UNIQUE (
                        source_id,
                        source_document_id,
                        document_checksum,
                        chunk_index
                    )
                )
                """,
            )
            await db.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS chunk_embeddings USING vec0(
                    chunk_id TEXT PRIMARY KEY,
                    embedding float[1024]
                )""",
            )

    async def save_bulk(self, *, chunks: list[DocumentChunk]) -> None:
        """Persist chunks and return number of inserted records."""
        if not chunks:
            return None

        async with self._db_manager.with_transaction() as db:
            res = await db.executemany(
                """
                INSERT OR IGNORE INTO document_chunks (
                    chunk_id,
                    source_id,
                    source_document_id,
                    document_checksum,
                    chunk_index,
                    start_char,
                    end_char,
                    chunk_text,
                    topics,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_checksum=excluded.document_checksum,
                    chunk_index=excluded.chunk_index,
                    start_char=excluded.start_char,
                    end_char=excluded.end_char,
                    chunk_text=excluded.chunk_text,
                    topics=excluded.topics,
                    metadata=excluded.metadata
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
                        json.dumps(chunk.topics) if chunk.topics is not None else None,
                        json.dumps(chunk.metadata) if chunk.metadata is not None else None,
                    )
                    for chunk in chunks
                ],
            )
        logger.info(f"Saved {res.rowcount} chunks to repository")

    async def save_embeddings(self, *, chunks: list[DocumentChunk]) -> None:
        """Persist one or more document embeddings."""
        if not chunks:
            return None

        async with self._db_manager.with_transaction() as db:
            embedding_tuples = [
                (chunk.chunk_id, serialize_f32(chunk.embedding))
                for chunk in chunks
                if chunk.embedding is not None
            ]
            placeholders = ",".join("?" for _ in embedding_tuples)
            async with db.execute(
                f"""
                SELECT chunk_id FROM chunk_embeddings
                WHERE chunk_id IN ({placeholders})
                """,
                [chunk_id for (chunk_id, _) in embedding_tuples],
            ) as curs:
                rows = await curs.fetchall()
                existing_chunk_ids = {row[0] for row in rows}
            to_update = [t for t in embedding_tuples if t[0] in existing_chunk_ids]
            to_insert = [t for t in embedding_tuples if t[0] not in existing_chunk_ids]
            if to_insert:
                await db.executemany(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, embedding)
                    VALUES (?, ?)
                    """,
                    to_insert,
                )
            if to_update:
                await db.executemany(
                    """
                    UPDATE chunk_embeddings
                    SET embedding = ?
                    WHERE chunk_id = ?
                    """,
                    [(t[1], t[0]) for t in to_update],
                )

    async def save(self, chunk: DocumentChunk) -> None:
        """Persist one chunk, inserting or updating as needed."""
        await self.save_bulk(chunks=[chunk])

    async def get_chunk(self, *, chunk_id: str) -> DocumentChunk | None:
        """Return one chunk by identifier or none when missing."""
        async with self._db_manager.with_transaction() as db:
            async with db.execute(
                """
                SELECT
                    chunk_id,
                    source_id,
                    source_document_id,
                    document_checksum,
                    chunk_index,
                    start_char,
                    end_char,
                    chunk_text,
                    topics,
                    metadata
                FROM document_chunks
                WHERE chunk_id = ?
                """,
                (chunk_id,),
            ) as curs:
                row = await curs.fetchone()
            async with db.execute(
                "SELECT chunk_id, embedding FROM chunk_embeddings",
            ) as curs:
                emb_row = await curs.fetchone()
        if row is None:
            return None
        embedding = None
        if emb_row is not None:
            embedding = deserialize_f32(emb_row[1])
        return DocumentChunk(
            chunk_id=row[0],
            source_id=row[1],
            source_document_id=row[2],
            document_checksum=row[3],
            chunk_index=row[4],
            start_char=row[5],
            end_char=row[6],
            chunk_text=row[7],
            embedding=embedding,
            topics=json.loads(row[8]) if row[8] else None,
            metadata=json.loads(row[9]) if row[9] else None,
        )

    async def list_chunks(
        self,
        *,
        chunk_ids: list[str] | None = None,
        source_id: str | None = None,
    ) -> list[DocumentChunk]:
        """List all chunks in the repository."""
        if chunk_ids is None:
            chunk_ids = []

        async with self._db_manager.with_transaction() as db:
            query = """
                SELECT
                    chunk_id,
                    source_id,
                    source_document_id,
                    document_checksum,
                    chunk_index,
                    start_char,
                    end_char,
                    chunk_text,
                    topics,
                    metadata
                FROM document_chunks
            """
            filters: list[str] = []
            vars: list[str] = []
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                filters.append(f"chunk_id IN ({placeholders})")
                vars.extend(chunk_ids)
            if source_id:
                filters.append("source_id = ?")
                vars.append(source_id)
            if filters:
                query += " WHERE " + " AND ".join(filters)
            async with db.execute(query, tuple(vars)) as curs:
                rows = await curs.fetchall()
            async with db.execute(
                "SELECT chunk_id, embedding FROM chunk_embeddings",
            ) as curs:
                emb_rows = await curs.fetchall()
        emb_map = {row[0]: deserialize_f32(row[1]) for row in emb_rows}
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
                topics=json.loads(row[8]) if row[8] else None,
                metadata=json.loads(row[9]) if row[9] else None,
                embedding=emb_map.get(row[0]),
            )
            for row in rows
        ]

    async def list_chunks_without_embedding(
        self,
        *,
        source_id: str | None = None,
    ) -> list[DocumentChunk]:
        """List all chunks without embeddings in the repository."""
        async with self._db_manager.with_transaction() as db:
            query = """
                SELECT
                    dc.chunk_id,
                    dc.source_id,
                    dc.source_document_id,
                    dc.document_checksum,
                    dc.chunk_index,
                    dc.start_char,
                    dc.end_char,
                    dc.chunk_text,
                    dc.topics,
                    dc.metadata
                FROM document_chunks dc
                LEFT JOIN chunk_embeddings ce ON dc.chunk_id = ce.chunk_id
                WHERE ce.chunk_id IS NULL
            """
            vars: list[str] = []
            if source_id:
                query += " AND dc.source_id = ?"
                vars.append(source_id)
            async with db.execute(query, tuple(vars)) as curs:
                rows = await curs.fetchall()
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
                topics=json.loads(row[8]) if row[8] else None,
                metadata=json.loads(row[9]) if row[9] else None,
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
