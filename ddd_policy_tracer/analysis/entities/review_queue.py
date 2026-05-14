"""SQLite review queue persistence for resolver outcomes."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from .models import EntityMention


@dataclass(frozen=True)
class ReviewQueueWriteReport:
    """Summarize one review queue enqueue operation."""

    queued_items: int
    appended_events: int


@dataclass(frozen=True)
class SQLiteReviewQueueRepository:
    """Persist unresolved resolver decisions in SQLite review queue tables."""

    sqlite_path: Path

    def __post_init__(self) -> None:
        """Create review queue schema on first repository usage."""
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_items (
                    mention_entity_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    document_checksum TEXT NOT NULL,
                    mention_text TEXT NOT NULL,
                    normalized_mention_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    decision_status TEXT NOT NULL,
                    decision_score REAL NOT NULL,
                    selected_candidate_key TEXT,
                    reason_codes_json TEXT NOT NULL,
                    top_candidates_json TEXT NOT NULL,
                    review_state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
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

    def enqueue_resolver_decisions(
        self,
        *,
        mentions: list[EntityMention],
    ) -> ReviewQueueWriteReport:
        """Persist unresolved resolver outputs and append audit events."""
        unresolved = [mention for mention in mentions if _is_unresolved(mention)]
        if not unresolved:
            return ReviewQueueWriteReport(queued_items=0, appended_events=0)

        queued_items = 0
        appended_events = 0
        timestamp = datetime.now(tz=UTC).isoformat()
        with self._connect() as connection:
            for mention in unresolved:
                metadata = dict(mention.metadata or {})
                reason_codes = metadata.get("reason_codes")
                top_candidates = metadata.get("top_candidates")
                selected_candidate_key = metadata.get("selected_candidate_key")
                decision_status = str(metadata.get("decision_status", "needs_review"))
                decision_score = float(metadata.get("decision_score", 0.0))

                cursor = connection.execute(
                    """
                    INSERT INTO review_items (
                        mention_entity_id,
                        chunk_id,
                        source_id,
                        source_document_id,
                        document_checksum,
                        mention_text,
                        normalized_mention_text,
                        entity_type,
                        decision_status,
                        decision_score,
                        selected_candidate_key,
                        reason_codes_json,
                        top_candidates_json,
                        review_state,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(mention_entity_id) DO UPDATE SET
                        decision_status=excluded.decision_status,
                        decision_score=excluded.decision_score,
                        selected_candidate_key=excluded.selected_candidate_key,
                        reason_codes_json=excluded.reason_codes_json,
                        top_candidates_json=excluded.top_candidates_json,
                        review_state=excluded.review_state,
                        updated_at=excluded.updated_at
                    """,
                    (
                        mention.entity_id,
                        mention.chunk_id,
                        mention.source_id,
                        mention.source_document_id,
                        mention.document_checksum,
                        mention.mention_text,
                        mention.normalized_mention_text,
                        mention.entity_type,
                        decision_status,
                        decision_score,
                        None if selected_candidate_key is None else str(selected_candidate_key),
                        json.dumps(reason_codes if isinstance(reason_codes, list) else [], ensure_ascii=True),
                        json.dumps(top_candidates if isinstance(top_candidates, list) else [], ensure_ascii=True),
                        "pending",
                        timestamp,
                        timestamp,
                    ),
                )
                queued_items += 1 if cursor.rowcount >= 1 else 0

                payload = {
                    "mention_entity_id": mention.entity_id,
                    "decision_status": decision_status,
                    "decision_score": decision_score,
                    "selected_candidate_key": selected_candidate_key,
                    "reason_codes": reason_codes if isinstance(reason_codes, list) else [],
                    "top_candidates": top_candidates if isinstance(top_candidates, list) else [],
                }
                event_hash = _build_event_hash(payload)
                event_cursor = connection.execute(
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
                        mention.entity_id,
                        "resolver_decision_upsert",
                        event_hash,
                        json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
                        timestamp,
                    ),
                )
                appended_events += max(0, event_cursor.rowcount)
        return ReviewQueueWriteReport(queued_items=queued_items, appended_events=appended_events)

    def list_review_items(self) -> list[dict[str, object]]:
        """Load current review queue item state for diagnostics/tests."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT mention_entity_id, decision_status, decision_score, review_state
                FROM review_items
                ORDER BY mention_entity_id ASC
                """,
            ).fetchall()
        return [
            {
                "mention_entity_id": str(row[0]),
                "decision_status": str(row[1]),
                "decision_score": float(row[2]),
                "review_state": str(row[3]),
            }
            for row in rows
        ]

    def count_review_events(self) -> int:
        """Count all persisted review events for audit verification."""
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM review_events").fetchone()
        return int(row[0])

    def _connect(self) -> sqlite3.Connection:
        """Open SQLite connection used by review queue operations."""
        return sqlite3.connect(self.sqlite_path)


def _is_unresolved(mention: EntityMention) -> bool:
    """Return true when mention status belongs in review queue."""
    metadata = mention.metadata or {}
    status = metadata.get("decision_status")
    if not isinstance(status, str):
        return False
    return status in {"needs_review", "new_candidate", "abstain"}


def _build_event_hash(payload: dict[str, object]) -> str:
    """Build deterministic event hash used for replay-safe inserts."""
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()
