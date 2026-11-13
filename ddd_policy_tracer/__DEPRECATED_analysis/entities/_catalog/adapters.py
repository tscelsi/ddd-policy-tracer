"""SQLite adapters for catalog entity and candidate repositories."""

import json
import struct

from ddd_policy_tracer.__DEPRECATED_analysis.entities._catalog.ports import (
    CatalogCandidate,
    CatalogEntity,
)
from ddd_policy_tracer.__DEPRECATED_analysis.entities.review_queue import ReviewState
from ddd_policy_tracer.utils.db_manager import with_transaction


def serialize_f32(vector: list[float]) -> bytes:
    """Serializes a list of floats into a compact "raw bytes" format."""
    return struct.pack("%sf" % len(vector), *vector)


def deserialize_f32(blob: bytes) -> list[float]:
    """Deserializes a blob of raw bytes back into a list of floats."""
    num_floats = len(blob) // 4
    return list(struct.unpack("%sf" % num_floats, blob))


class SQLiteCatalogEntityRepository:
    """SQLite-backed repository for catalog entities."""

    async def init(self):
        await self._ensure_schema()

    async def _ensure_schema(self):
        async with with_transaction() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog_entities (
                    canonical_entity_key TEXT PRIMARY KEY,
                    canonical_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    description TEXT,
                    aliases_json TEXT,
                    contexts_json TEXT
                )
                """,
            )
            await db.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS catalog_entity_embeddings USING vec0(
                    canonical_entity_key TEXT PRIMARY KEY,
                    embedding float[1024]
                )""",
            )

    async def list_entities(self) -> list[CatalogEntity]:
        """Return all catalog entities as entity mentions for resolution."""
        async with with_transaction() as db:
            async with db.execute(
                "SELECT canonical_entity_key, canonical_name, entity_type, description, aliases_json, contexts_json FROM catalog_entities",
            ) as curs:
                rows = await curs.fetchall()
            async with db.execute(
                "SELECT canonical_entity_key, embedding FROM catalog_entity_embeddings",
            ) as curs:
                emb_rows = await curs.fetchall()
        embedding_map = {row[0]: deserialize_f32(row[1]) for row in emb_rows}
        entities = []
        for row in rows:
            entity = CatalogEntity(
                canonical_entity_key=row[0],
                canonical_name=row[1],
                entity_type=row[2],
                description=row[3],
                aliases=json.loads(row[4]) if row[4] else [],
                contexts=json.loads(row[5]) if row[5] else [],
                embedding=embedding_map.get(row[0]),
            )
            entities.append(entity)
        return entities

    async def get_catalog_entity_by_key(self, *, key: str) -> CatalogEntity | None:
        async with with_transaction() as db:
            async with db.execute(
                "SELECT canonical_entity_key, canonical_name, entity_type, description, aliases_json, contexts_json FROM catalog_entities WHERE canonical_entity_key = ?",
                (key,),
            ) as curs:
                row = await curs.fetchone()
        if not row:
            return None
        return CatalogEntity(
            canonical_entity_key=row[0],
            canonical_name=row[1],
            entity_type=row[2],
            description=row[3],
            aliases=json.loads(row[4]) if row[4] else [],
            contexts=json.loads(row[5]) if row[5] else [],
        )

    async def get_catalog_entities_by_keys(self, *, keys: list[str]) -> list[CatalogEntity]:
        """Return multiple catalog entities by their canonical keys."""
        async with with_transaction() as db:
            async with db.execute(
                "SELECT canonical_entity_key, canonical_name, entity_type, description, aliases_json, contexts_json FROM catalog_entities WHERE canonical_entity_key IN ({})".format(
                    ",".join("?" for _ in keys),
                ),
                keys,
            ) as curs:
                rows = await curs.fetchall()
        if not rows:
            return []
        entities: list[CatalogEntity] = []
        for row in rows:
            entity = CatalogEntity(
                canonical_entity_key=row[0],
                canonical_name=row[1],
                entity_type=row[2],
                description=row[3],
                aliases=json.loads(row[4]) if row[4] else [],
                contexts=json.loads(row[5]) if row[5] else [],
            )
            entities.append(entity)
        return entities

    async def save(self, entity: CatalogEntity) -> None:
        async with with_transaction() as db:
            await db.execute(
                """
                INSERT INTO catalog_entities (
                    canonical_entity_key, canonical_name, entity_type, description, aliases_json, contexts_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_entity_key) DO UPDATE SET
                    canonical_name=excluded.canonical_name,
                    entity_type=excluded.entity_type,
                    description=excluded.description,
                    aliases_json=excluded.aliases_json,
                    contexts_json=excluded.contexts_json
                """,
                (
                    entity.canonical_entity_key,
                    entity.canonical_name,
                    entity.entity_type,
                    getattr(entity, "description", None),
                    json.dumps(getattr(entity, "aliases", []) or []),
                    json.dumps(getattr(entity, "contexts", []) or []),
                ),
            )
            if entity.embedding is not None:
                serialized_embedding = serialize_f32(entity.embedding)
                await db.execute(
                    """
                    INSERT OR REPLACE INTO catalog_entity_embeddings (canonical_entity_key, embedding)
                    VALUES (?, ?)
                    """,
                    (entity.canonical_entity_key, serialized_embedding),
                )

    async def get_similar(
        self,
        *,
        query: list[float],
        top_k: int = 5,
    ) -> tuple[list[CatalogEntity], list[float]]:
        """Return the most similar catalog entities to the provided embedding."""
        async with with_transaction() as db:
            query_embedding = serialize_f32(query)
            async with await db.execute(
                """
                SELECT canonical_entity_key, distance
                FROM catalog_entity_embeddings
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (query_embedding, top_k),
            ) as curs:
                rows = await curs.fetchall()
        keys = [row[0] for row in rows]
        entities = await self.get_catalog_entities_by_keys(keys=keys)
        return entities, [row[1] for row in rows]


class SQLiteCatalogCandidateRepository:
    """SQLite-backed repository for catalog candidates."""

    def __init__(self) -> None:
        """Initialize the repository with the path to the SQLite database."""
        self._ensure_schema()

    async def _ensure_schema(self) -> None:
        """Create the catalog_candidates table if it doesn't exist."""
        async with with_transaction() as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog_candidates (
                    entity_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    document_checksum TEXT NOT NULL,
                    start_char INTEGER NOT NULL,
                    end_char INTEGER NOT NULL,
                    mention_text TEXT NOT NULL,
                    normalized_mention_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    extractor_version TEXT NOT NULL,
                    canonical_entity_key TEXT,
                    metadata_json TEXT,
                    review_state TEXT NOT NULL
                )
                """,
            )

    async def list_candidates(self, status: ReviewState | None = None) -> list[CatalogCandidate]:
        """Return catalog candidates with optional filtering by review state."""
        query = "SELECT entity_id, chunk_id, source_id, source_document_id, document_checksum, start_char, end_char, mention_text, normalized_mention_text, entity_type, confidence, extractor_version, canonical_entity_key, metadata_json, review_state FROM catalog_candidates"
        params = ()
        if status is not None:
            query += " WHERE review_state = ?"
            params = (status,)
        async with with_transaction() as db:
            async with db.execute(query, params) as curs:
                rows = await curs.fetchall()
        candidates = []
        for row in rows:
            candidate = CatalogCandidate(
                entity_id=row[0],
                chunk_id=row[1],
                source_id=row[2],
                source_document_id=row[3],
                document_checksum=row[4],
                start_char=row[5],
                end_char=row[6],
                mention_text=row[7],
                normalized_mention_text=row[8],
                entity_type=row[9],
                confidence=row[10],
                extractor_version=row[11],
                canonical_entity_key=row[12],
                metadata=json.loads(row[13]) if row[13] else None,
                review_state=row[14],
            )
            candidates.append(candidate)
        return candidates

    async def list_candidates_by_entity_ids(
        self,
        *,
        entity_ids: list[str],
    ) -> list[CatalogCandidate]:
        """Return candidates linked to any of the provided entity mention identities."""
        if not entity_ids:
            return []
        placeholders = ",".join(["?"] * len(entity_ids))
        query = (
            f"SELECT entity_id, chunk_id, source_id, source_document_id, document_checksum, start_char, end_char, mention_text, normalized_mention_text, entity_type, confidence, extractor_version, canonical_entity_key, metadata_json, review_state "
            f"FROM catalog_candidates WHERE entity_id IN ({placeholders})"
        )
        async with with_transaction() as db:
            async with db.execute(query, entity_ids) as curs:
                rows = await curs.fetchall()
        candidates = []
        for row in rows:
            candidate = CatalogCandidate(
                entity_id=row[0],
                chunk_id=row[1],
                source_id=row[2],
                source_document_id=row[3],
                document_checksum=row[4],
                start_char=row[5],
                end_char=row[6],
                mention_text=row[7],
                normalized_mention_text=row[8],
                entity_type=row[9],
                confidence=row[10],
                extractor_version=row[11],
                canonical_entity_key=row[12],
                metadata=json.loads(row[13]) if row[13] else None,
                review_state=row[14],
            )
            candidates.append(candidate)
        return candidates

    async def save_bulk(self, candidates: list[CatalogCandidate]) -> None:
        """Add or update many catalog candidates in one batch operation."""
        if not candidates:
            return
        async with with_transaction() as db:
            await db.executemany(
                """
                INSERT INTO catalog_candidates (
                    entity_id, chunk_id, source_id, source_document_id, document_checksum, start_char, end_char, mention_text, normalized_mention_text, entity_type, confidence, extractor_version, canonical_entity_key, metadata_json, review_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    chunk_id=excluded.chunk_id,
                    source_id=excluded.source_id,
                    source_document_id=excluded.source_document_id,
                    document_checksum=excluded.document_checksum,
                    start_char=excluded.start_char,
                    end_char=excluded.end_char,
                    mention_text=excluded.mention_text,
                    normalized_mention_text=excluded.normalized_mention_text,
                    entity_type=excluded.entity_type,
                    confidence=excluded.confidence,
                    extractor_version=excluded.extractor_version,
                    canonical_entity_key=excluded.canonical_entity_key,
                    metadata_json=excluded.metadata_json,
                    review_state=excluded.review_state
                """,
                [
                    (
                        c.entity_id,
                        c.chunk_id,
                        c.source_id,
                        c.source_document_id,
                        c.document_checksum,
                        c.start_char,
                        c.end_char,
                        c.mention_text,
                        c.normalized_mention_text,
                        c.entity_type,
                        c.confidence,
                        c.extractor_version,
                        c.canonical_entity_key,
                        json.dumps(c.metadata) if c.metadata is not None else None,
                        c.review_state,
                    )
                    for c in candidates
                ],
            )

    async def get_candidate_by_entity_id(self, *, entity_id: str) -> CatalogCandidate | None:
        """Return one candidate by entity mention identity or none when missing."""
        async with with_transaction() as db:
            async with db.execute(
                "SELECT entity_id, chunk_id, source_id, source_document_id, document_checksum, start_char, end_char, mention_text, normalized_mention_text, entity_type, confidence, extractor_version, canonical_entity_key, metadata_json, review_state FROM catalog_candidates WHERE entity_id = ?",
                (entity_id,),
            ) as curs:
                row = await curs.fetchone()
        if not row:
            return None
        return CatalogCandidate(
            entity_id=row[0],
            chunk_id=row[1],
            source_id=row[2],
            source_document_id=row[3],
            document_checksum=row[4],
            start_char=row[5],
            end_char=row[6],
            mention_text=row[7],
            normalized_mention_text=row[8],
            entity_type=row[9],
            confidence=row[10],
            extractor_version=row[11],
            canonical_entity_key=row[12],
            metadata=json.loads(row[13]) if row[13] else None,
            review_state=row[14],
        )

    async def save(self, candidate: CatalogCandidate) -> None:
        """Add or update a catalog candidate."""
        async with with_transaction() as db:
            await db.execute(
                """
                INSERT INTO catalog_candidates (
                    entity_id, chunk_id, source_id, source_document_id, document_checksum, start_char, end_char, mention_text, normalized_mention_text, entity_type, confidence, extractor_version, canonical_entity_key, metadata_json, review_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    chunk_id=excluded.chunk_id,
                    source_id=excluded.source_id,
                    source_document_id=excluded.source_document_id,
                    document_checksum=excluded.document_checksum,
                    start_char=excluded.start_char,
                    end_char=excluded.end_char,
                    mention_text=excluded.mention_text,
                    normalized_mention_text=excluded.normalized_mention_text,
                    entity_type=excluded.entity_type,
                    confidence=excluded.confidence,
                    extractor_version=excluded.extractor_version,
                    canonical_entity_key=excluded.canonical_entity_key,
                    metadata_json=excluded.metadata_json,
                    review_state=excluded.review_state
                """,
                (
                    candidate.entity_id,
                    candidate.chunk_id,
                    candidate.source_id,
                    candidate.source_document_id,
                    candidate.document_checksum,
                    candidate.start_char,
                    candidate.end_char,
                    candidate.mention_text,
                    candidate.normalized_mention_text,
                    candidate.entity_type,
                    candidate.confidence,
                    candidate.extractor_version,
                    candidate.canonical_entity_key,
                    json.dumps(candidate.metadata) if candidate.metadata is not None else None,
                    candidate.review_state,
                ),
            )
