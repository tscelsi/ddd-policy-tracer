"""SQLite review queue persistence and decision workflows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

from .models import CatalogCandidate, EntityMention

ReviewState = Literal["pending", "approved", "rejected", "promoted"]


@dataclass(frozen=True)
class ReviewQueueWriteReport:
    """Summarize one review queue enqueue operation."""

    queued_items: int
    appended_events: int


@dataclass(frozen=True)
class ReviewDecisionReport:
    """Summarize one explicit review decision update."""

    mention_entity_id: str
    review_state: ReviewState
    updated_rows: int


@dataclass(frozen=True)
class PromotionCandidate:
    """Represent one approved review item ready for catalog promotion."""

    mention_entity_id: str
    canonical_entity_key: str
    canonical_name: str
    entity_type: str
    aliases: list[str]
    decision_score: float
    source: str


@dataclass(frozen=True)
class SQLiteReviewQueueRepository:
    """Persist unresolved resolver decisions and review outcomes in SQLite."""

    sqlite_path: Path

    def __post_init__(self) -> None:
        """Ensure review queue schema is available before reads/writes."""
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
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
                    metadata_json TEXT
                )
                """,
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mention_entity_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """,
            )
            _ensure_column(
                connection=connection,
                table_name="catalog_candidates",
                column_name="start_char",
                column_type="INTEGER NOT NULL DEFAULT 0",
            )
            _ensure_column(
                connection=connection,
                table_name="catalog_candidates",
                column_name="end_char",
                column_type="INTEGER NOT NULL DEFAULT 0",
            )

    def enqueue_resolver_decisions(
        self,
        *,
        mentions: list[EntityMention],
    ) -> ReviewQueueWriteReport:
        """Persist unresolved resolver outputs and append deduplicated events using new schema."""
        unresolved = [mention for mention in mentions if _is_unresolved(mention)]
        if not unresolved:
            return ReviewQueueWriteReport(queued_items=0, appended_events=0)

        queued_items = 0
        appended_events = 0
        timestamp = _now_iso()
        with self._connect() as connection:
            for mention in unresolved:
                # Prepare metadata JSON
                candidate = CatalogCandidate.from_entity_mention(mention)
                metadata_json = (
                    json.dumps(candidate.metadata, ensure_ascii=True)
                    if candidate.metadata is not None
                    else None
                )

                # Insert or update using new schema fields
                cursor = connection.execute(
                    """
                    INSERT INTO catalog_candidates (
                        entity_id,
                        chunk_id,
                        source_id,
                        source_document_id,
                        document_checksum,
                        start_char,
                        end_char,
                        mention_text,
                        normalized_mention_text,
                        entity_type,
                        confidence,
                        extractor_version,
                        canonical_entity_key,
                        metadata_json,
                        review_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_id) DO UPDATE SET
                        start_char=excluded.start_char,
                        end_char=excluded.end_char,
                        confidence=excluded.confidence,
                        extractor_version=excluded.extractor_version,
                        canonical_entity_key=excluded.canonical_entity_key,
                        metadata_json=excluded.metadata_json,
                        review_state=CASE
                            WHEN catalog_candidates.review_state IN ('approved', 'promoted')
                            THEN catalog_candidates.review_state
                            ELSE excluded.review_state
                        END
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
                        metadata_json,
                        candidate.review_state,
                    ),
                )
                queued_items += max(0, cursor.rowcount)

                payload = {
                    "entity_id": mention.entity_id,
                    "metadata": mention.metadata,
                }
                appended_events += self._append_event(
                    connection=connection,
                    mention_entity_id=mention.entity_id,
                    event_type="resolver_decision_upsert",
                    payload=payload,
                    created_at=timestamp,
                )
        return ReviewQueueWriteReport(queued_items=queued_items, appended_events=appended_events)

    def list_review_items(self, *, state: ReviewState | None = None) -> list[CatalogCandidate]:
        """Return review queue items, optionally filtered by review state."""
        with self._connect() as connection:
            if state is None:
                rows = connection.execute(
                    """
                    SELECT entity_id,
                           chunk_id,
                           source_id,
                           source_document_id,
                           document_checksum,
                           start_char,
                           end_char,
                           mention_text,
                           normalized_mention_text,
                           entity_type,
                           confidence,
                           extractor_version,
                           canonical_entity_key,
                           metadata_json,
                           review_state
                    FROM catalog_candidates
                    ORDER BY entity_id ASC
                    """,
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT entity_id,
                           chunk_id,
                           source_id,
                           source_document_id,
                           document_checksum,
                           start_char,
                           end_char,
                           mention_text,
                           normalized_mention_text,
                           entity_type,
                           confidence,
                           extractor_version,
                           canonical_entity_key,
                           metadata_json,
                           review_state
                    FROM catalog_candidates
                    WHERE review_state = ?
                    ORDER BY entity_id ASC
                    """,
                    (state,),
                ).fetchall()

        items: list[CatalogCandidate] = []
        for row in rows:
            metadata = None
            if row[13] is not None:
                try:
                    metadata = json.loads(row[13])
                except json.JSONDecodeError:
                    pass
            items.append(
                CatalogCandidate(
                    entity_id=str(row[0]),
                    chunk_id=str(row[1]),
                    source_id=str(row[2]),
                    source_document_id=str(row[3]),
                    document_checksum=str(row[4]),
                    start_char=int(row[5]),
                    end_char=int(row[6]),
                    mention_text=str(row[7]),
                    normalized_mention_text=str(row[8]),
                    entity_type=str(row[9]),
                    confidence=float(row[10]),
                    extractor_version=str(row[11]),
                    canonical_entity_key=None if row[12] is None else str(row[12]),
                    metadata=metadata,
                    review_state=str(row[14]) if row[14] is not None else "pending",
                ),
            )
        return items

    def decide_item(
        self,
        *,
        mention_entity_id: str,
        decision: Literal["approved", "rejected"],
        canonical_entity_key: str | None = None,
        canonical_name: str | None = None,
        entity_type: str | None = None,
        aliases: list[str] | None = None,
        description: str | None = None,
    ) -> ReviewDecisionReport:
        """Apply one explicit review decision and update canonical fields."""
        timestamp = _now_iso()
        normalized_aliases = _normalize_aliases(aliases or [])
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT mention_entity_id, canonical_entity_key, canonical_name, entity_type
                FROM catalog_candidates
                WHERE mention_entity_id = ?
                """,
                (mention_entity_id,),
            ).fetchone()
            if row is None:
                raise ValueError("review item not found")

            existing_key = None if row[1] is None else str(row[1])
            existing_name = None if row[2] is None else str(row[2])
            existing_entity_type = str(row[4])

            resolved_key = canonical_entity_key or existing_key
            resolved_name = canonical_name or existing_name
            if resolved_name and resolved_name not in resolved_aliases:
                resolved_aliases = _normalize_aliases([resolved_name, *resolved_aliases])
            resolved_entity_type = entity_type or existing_entity_type

            if decision == "approved":
                if resolved_key is None or not resolved_key.strip():
                    raise ValueError("approved decision requires canonical_entity_key")
                if resolved_name is None or not resolved_name.strip():
                    raise ValueError("approved decision requires canonical_name")
                if not resolved_entity_type.strip():
                    raise ValueError("approved decision requires entity_type")

            cursor = connection.execute(
                """
                UPDATE review_items
                SET review_state = ?,
                    canonical_entity_key = ?,
                    canonical_name = ?,
                    canonical_aliases_json = ?,
                    entity_type = ?,
                    updated_at = ?
                WHERE mention_entity_id = ?
                """,
                (
                    decision,
                    resolved_key,
                    resolved_name,
                    json.dumps(resolved_aliases, ensure_ascii=True),
                    resolved_entity_type,
                    timestamp,
                    mention_entity_id,
                ),
            )
            self._append_event(
                connection=connection,
                mention_entity_id=mention_entity_id,
                event_type="review_decision",
                payload={
                    "decision": decision,
                    "canonical_entity_key": resolved_key,
                    "canonical_name": resolved_name,
                    "entity_type": resolved_entity_type,
                    "aliases": resolved_aliases,
                },
                created_at=timestamp,
            )
        return ReviewDecisionReport(
            mention_entity_id=mention_entity_id,
            review_state=decision,
            updated_rows=max(0, cursor.rowcount),
        )

    def list_promotion_candidates(self) -> list[PromotionCandidate]:
        """Return approved or auto-approved items eligible for promotion."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT mention_entity_id,
                       canonical_entity_key,
                       canonical_name,
                       entity_type,
                       canonical_aliases_json,
                       decision_score,
                       review_state
                FROM review_items
                WHERE review_state IN ('approved')
                  AND canonical_entity_key IS NOT NULL
                  AND canonical_name IS NOT NULL
                ORDER BY mention_entity_id ASC
                """,
            ).fetchall()
        candidates: list[PromotionCandidate] = []
        for row in rows:
            key = str(row[1]).strip()
            name = str(row[2]).strip()
            entity_type = str(row[3]).strip()
            if not key or not name or not entity_type:
                continue
            candidates.append(
                PromotionCandidate(
                    mention_entity_id=str(row[0]),
                    canonical_entity_key=key,
                    canonical_name=name,
                    entity_type=entity_type,
                    aliases=_alias_list_from_cell(row[4]),
                    decision_score=float(row[5]),
                    source="review_queue",
                ),
            )
        return candidates

    def mark_promoted(self, *, mention_entity_ids: list[str]) -> int:
        """Mark selected review items as promoted and append audit events."""
        if not mention_entity_ids:
            return 0
        timestamp = _now_iso()
        updated = 0
        with self._connect() as connection:
            for mention_entity_id in mention_entity_ids:
                cursor = connection.execute(
                    """
                    UPDATE review_items
                    SET review_state = 'promoted',
                        promoted_at = ?,
                        updated_at = ?
                    WHERE mention_entity_id = ?
                      AND review_state IN ('approved')
                    """,
                    (timestamp, timestamp, mention_entity_id),
                )
                if cursor.rowcount <= 0:
                    continue
                updated += cursor.rowcount
                self._append_event(
                    connection=connection,
                    mention_entity_id=mention_entity_id,
                    event_type="review_promoted",
                    payload={"promoted_at": timestamp},
                    created_at=timestamp,
                )
        return updated

    def count_review_events(self) -> int:
        """Return total persisted review events for audit verification."""
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM review_events").fetchone()
        return int(row[0])

    def _append_event(
        self,
        *,
        connection: sqlite3.Connection,
        mention_entity_id: str,
        event_type: str,
        payload: dict[str, object],
        created_at: str,
    ) -> int:
        """Append one deduplicated review event row."""
        event_hash = _build_event_hash(
            mention_entity_id=mention_entity_id,
            event_type=event_type,
            payload=payload,
        )
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO review_events (
                mention_entity_id,
                event_type,
                event_hash,
                payload_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                mention_entity_id,
                event_type,
                event_hash,
                json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                created_at,
            ),
        )
        return max(0, cursor.rowcount)

    def _connect(self) -> sqlite3.Connection:
        """Open one SQLite connection for review queue operations."""
        return sqlite3.connect(self.sqlite_path)


def _is_unresolved(mention: EntityMention) -> bool:
    """Return true when mention should be queued for review."""
    metadata = mention.metadata or {}
    status = metadata.get("decision_status")
    eligible = metadata.get("catalog_eligible")
    return (
        isinstance(status, str)
        and status in {"needs_review", "new_candidate", "abstain"}
        and eligible is not False
    )


def _normalize_json_list(value: object) -> list[object]:
    """Return a JSON-serializable list from one untyped value."""
    if not isinstance(value, list):
        return []
    normalized: list[object] = []
    for item in value:
        if isinstance(item, (str, int, float, bool)) or item is None:
            normalized.append(item)
            continue
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def _json_list_from_cell(cell: object) -> list[object]:
    """Decode one JSON list cell into raw list values."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        parsed = json.loads(cell)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _alias_list_from_cell(cell: object) -> list[str]:
    """Decode one JSON list cell into normalized string aliases."""
    parsed = _json_list_from_cell(cell)
    normalized = [
        str(item).strip() for item in parsed if isinstance(item, str) and str(item).strip()
    ]
    return _normalize_aliases(normalized)


def _as_float(value: object, *, fallback: float) -> float:
    """Convert one numeric-like value to float with fallback."""
    if isinstance(value, (int, float)):
        return float(value)
    return fallback


def _normalize_aliases(values: list[str]) -> list[str]:
    """Normalize aliases to case-insensitive deduplicated list."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(value.split())
        if not cleaned:
            continue
        marker = cleaned.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        normalized.append(cleaned)
    return normalized


def _build_event_hash(
    *,
    mention_entity_id: str,
    event_type: str,
    payload: dict[str, object],
) -> str:
    """Build deterministic hash used for event deduplication."""
    serialized = json.dumps(
        {
            "mention_entity_id": mention_entity_id,
            "event_type": event_type,
            "payload": payload,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(tz=UTC).isoformat()


def _ensure_column(
    *,
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    """Add one SQLite column when migrating existing review tables."""
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing = {str(row[1]) for row in rows}
    if column_name in existing:
        return
    connection.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}",
    )
