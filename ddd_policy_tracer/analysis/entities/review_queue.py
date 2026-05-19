"""SQLite review queue persistence and decision workflows."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

from .models import EntityMention

ReviewState = Literal["pending", "approved", "rejected", "auto_approved", "promoted"]


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
class AutoApprovalReport:
    """Summarize automatic approval actions over pending review items."""

    considered_items: int
    auto_approved_items: int


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
                CREATE TABLE IF NOT EXISTS review_items (
                    mention_entity_id TEXT PRIMARY KEY,
                    chunk_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    source_document_id TEXT NOT NULL,
                    document_checksum TEXT NOT NULL,
                    start_char INTEGER NOT NULL DEFAULT 0,
                    end_char INTEGER NOT NULL DEFAULT 0,
                    mention_text TEXT NOT NULL,
                    normalized_mention_text TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    decision_status TEXT NOT NULL,
                    decision_score REAL NOT NULL,
                    selected_candidate_key TEXT,
                    reason_codes_json TEXT NOT NULL,
                    top_candidates_json TEXT NOT NULL,
                    review_state TEXT NOT NULL,
                    canonical_entity_key TEXT,
                    canonical_name TEXT,
                    canonical_aliases_json TEXT NOT NULL,
                    promoted_at TEXT,
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
            _ensure_column(
                connection=connection,
                table_name="review_items",
                column_name="start_char",
                column_type="INTEGER NOT NULL DEFAULT 0",
            )
            _ensure_column(
                connection=connection,
                table_name="review_items",
                column_name="end_char",
                column_type="INTEGER NOT NULL DEFAULT 0",
            )

    def enqueue_resolver_decisions(
        self,
        *,
        mentions: list[EntityMention],
    ) -> ReviewQueueWriteReport:
        """Persist unresolved resolver outputs and append deduplicated events."""
        unresolved = [mention for mention in mentions if _is_unresolved(mention)]
        if not unresolved:
            return ReviewQueueWriteReport(queued_items=0, appended_events=0)

        queued_items = 0
        appended_events = 0
        timestamp = _now_iso()
        with self._connect() as connection:
            for mention in unresolved:
                metadata = dict(mention.metadata or {})
                reason_codes = _normalize_json_list(metadata.get("reason_codes"))
                top_candidates = _normalize_json_list(metadata.get("top_candidates"))
                candidate_links = _normalize_json_list(metadata.get("candidate_links"))
                selected_candidate_key = metadata.get("selected_candidate_key")
                decision_status = str(metadata.get("decision_status", "needs_review"))
                decision_score = _as_float(metadata.get("decision_score"), fallback=0.0)

                canonical_key = None
                canonical_name = None
                aliases_json = "[]"
                top_for_defaults = candidate_links[0] if candidate_links else (top_candidates[0] if top_candidates else None)
                if isinstance(top_for_defaults, dict):
                    top = top_for_defaults
                    if isinstance(top, dict):
                        candidate_key = top.get("canonical_entity_key")
                        candidate_name = top.get("canonical_name")
                        if isinstance(candidate_key, str) and candidate_key.strip():
                            canonical_key = candidate_key.strip()
                        if isinstance(candidate_name, str) and candidate_name.strip():
                            canonical_name = candidate_name.strip()
                            aliases_json = json.dumps([canonical_name], ensure_ascii=True)

                cursor = connection.execute(
                    """
                    INSERT INTO review_items (
                        mention_entity_id,
                        chunk_id,
                        source_id,
                        source_document_id,
                        document_checksum,
                        start_char,
                        end_char,
                        mention_text,
                        normalized_mention_text,
                        entity_type,
                        decision_status,
                        decision_score,
                        selected_candidate_key,
                        reason_codes_json,
                        top_candidates_json,
                        review_state,
                        canonical_entity_key,
                        canonical_name,
                        canonical_aliases_json,
                        promoted_at,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(mention_entity_id) DO UPDATE SET
                        start_char=excluded.start_char,
                        end_char=excluded.end_char,
                        decision_status=excluded.decision_status,
                        decision_score=excluded.decision_score,
                        selected_candidate_key=excluded.selected_candidate_key,
                        reason_codes_json=excluded.reason_codes_json,
                        top_candidates_json=excluded.top_candidates_json,
                        review_state=CASE
                            WHEN review_items.review_state IN ('approved', 'auto_approved', 'promoted')
                            THEN review_items.review_state
                            ELSE excluded.review_state
                        END,
                        canonical_entity_key=COALESCE(review_items.canonical_entity_key, excluded.canonical_entity_key),
                        canonical_name=COALESCE(review_items.canonical_name, excluded.canonical_name),
                        canonical_aliases_json=CASE
                            WHEN review_items.canonical_aliases_json = '[]'
                            THEN excluded.canonical_aliases_json
                            ELSE review_items.canonical_aliases_json
                        END,
                        updated_at=excluded.updated_at
                    """,
                    (
                        mention.entity_id,
                        mention.chunk_id,
                        mention.source_id,
                        mention.source_document_id,
                        mention.document_checksum,
                        mention.start_char,
                        mention.end_char,
                        mention.mention_text,
                        mention.normalized_mention_text,
                        mention.entity_type,
                        decision_status,
                        decision_score,
                        None if selected_candidate_key is None else str(selected_candidate_key),
                        json.dumps(reason_codes, ensure_ascii=True),
                        json.dumps(top_candidates, ensure_ascii=True),
                        "pending",
                        canonical_key,
                        canonical_name,
                        aliases_json,
                        None,
                        timestamp,
                        timestamp,
                    ),
                )
                queued_items += max(0, cursor.rowcount)

                payload = {
                    "mention_entity_id": mention.entity_id,
                    "decision_status": decision_status,
                    "decision_score": decision_score,
                    "selected_candidate_key": selected_candidate_key,
                    "reason_codes": reason_codes,
                    "candidate_links": candidate_links,
                    "top_candidates": top_candidates,
                }
                appended_events += self._append_event(
                    connection=connection,
                    mention_entity_id=mention.entity_id,
                    event_type="resolver_decision_upsert",
                    payload=payload,
                    created_at=timestamp,
                )
        return ReviewQueueWriteReport(queued_items=queued_items, appended_events=appended_events)

    def list_review_items(self, *, state: ReviewState | None = None) -> list[dict[str, object]]:
        """Return review queue items, optionally filtered by review state."""
        query = (
            "SELECT mention_entity_id, decision_status, decision_score, review_state, "
            "canonical_entity_key, canonical_name, canonical_aliases_json, selected_candidate_key "
            "FROM review_items"
        )
        params: tuple[object, ...] = ()
        if state is not None:
            query += " WHERE review_state = ?"
            params = (state,)
        query += " ORDER BY mention_entity_id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "mention_entity_id": str(row[0]),
                "decision_status": str(row[1]),
                "decision_score": float(row[2]),
                "review_state": str(row[3]),
                "canonical_entity_key": None if row[4] is None else str(row[4]),
                "canonical_name": None if row[5] is None else str(row[5]),
                "canonical_aliases": _alias_list_from_cell(row[6]),
                "selected_candidate_key": None if row[7] is None else str(row[7]),
            }
            for row in rows
        ]

    def list_pending_curation_items(self, *, limit: int = 50) -> list[dict[str, object]]:
        """Return pending catalog-eligible review items for interactive curation."""
        if limit <= 0:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT mention_entity_id,
                       chunk_id,
                       source_id,
                       source_document_id,
                       document_checksum,
                       start_char,
                       end_char,
                       mention_text,
                       normalized_mention_text,
                       entity_type,
                       decision_status,
                       decision_score,
                       selected_candidate_key,
                       reason_codes_json,
                       top_candidates_json,
                       review_state
                FROM review_items
                WHERE review_state = 'pending'
                ORDER BY mention_entity_id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        items: list[dict[str, object]] = []
        for row in rows:
            items.append(
                {
                    "mention_entity_id": str(row[0]),
                    "chunk_id": str(row[1]),
                    "source_id": str(row[2]),
                    "source_document_id": str(row[3]),
                    "document_checksum": str(row[4]),
                    "start_char": int(row[5]),
                    "end_char": int(row[6]),
                    "mention_text": str(row[7]),
                    "normalized_mention_text": str(row[8]),
                    "entity_type": str(row[9]),
                    "decision_status": str(row[10]),
                    "decision_score": float(row[11]),
                    "selected_candidate_key": None if row[12] is None else str(row[12]),
                    "reason_codes": _json_list_from_cell(row[13]),
                    "top_candidates": _json_list_from_cell(row[14]),
                    "candidate_links": _json_list_from_cell(row[14]),
                    "review_state": str(row[15]),
                },
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
    ) -> ReviewDecisionReport:
        """Apply one explicit review decision and update canonical fields."""
        timestamp = _now_iso()
        normalized_aliases = _normalize_aliases(aliases or [])
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT mention_entity_id, canonical_entity_key, canonical_name,
                       canonical_aliases_json, entity_type
                FROM review_items
                WHERE mention_entity_id = ?
                """,
                (mention_entity_id,),
            ).fetchone()
            if row is None:
                raise ValueError("review item not found")

            existing_key = None if row[1] is None else str(row[1])
            existing_name = None if row[2] is None else str(row[2])
            existing_aliases = _alias_list_from_cell(row[3])
            existing_entity_type = str(row[4])

            resolved_key = canonical_entity_key or existing_key
            resolved_name = canonical_name or existing_name
            resolved_aliases = normalized_aliases or existing_aliases
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

    def auto_approve_obvious(
        self,
        *,
        min_score: float,
        min_margin: float,
    ) -> AutoApprovalReport:
        """Automatically approve high-confidence low-ambiguity pending items."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT mention_entity_id,
                       decision_status,
                       decision_score,
                       top_candidates_json,
                       entity_type,
                       review_state
                FROM review_items
                ORDER BY mention_entity_id ASC
                """,
            ).fetchall()

            considered = 0
            approved = 0
            for row in rows:
                mention_entity_id = str(row[0])
                decision_status = str(row[1])
                decision_score = float(row[2])
                top_candidates = _json_list_from_cell(row[3])
                mention_entity_type = str(row[4])
                review_state = str(row[5])
                if review_state not in {"pending"}:
                    continue
                considered += 1
                if not _passes_auto_approval_gates(
                    decision_status=decision_status,
                    decision_score=decision_score,
                    top_candidates=top_candidates,
                    mention_entity_type=mention_entity_type,
                    min_score=min_score,
                    min_margin=min_margin,
                ):
                    continue

                top_candidate = top_candidates[0]
                canonical_key = str(top_candidate.get("canonical_entity_key", "")).strip()
                canonical_name = str(top_candidate.get("canonical_name", "")).strip()
                entity_type = str(top_candidate.get("entity_type", mention_entity_type)).strip()
                if not canonical_key or not canonical_name or not entity_type:
                    continue

                timestamp = _now_iso()
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
                        "auto_approved",
                        canonical_key,
                        canonical_name,
                        json.dumps([canonical_name], ensure_ascii=True),
                        entity_type,
                        timestamp,
                        mention_entity_id,
                    ),
                )
                if cursor.rowcount <= 0:
                    continue
                approved += 1
                self._append_event(
                    connection=connection,
                    mention_entity_id=mention_entity_id,
                    event_type="review_auto_approved",
                    payload={
                        "decision_status": decision_status,
                        "decision_score": decision_score,
                        "canonical_entity_key": canonical_key,
                        "canonical_name": canonical_name,
                        "entity_type": entity_type,
                        "min_score": min_score,
                        "min_margin": min_margin,
                    },
                    created_at=timestamp,
                )
        return AutoApprovalReport(considered_items=considered, auto_approved_items=approved)

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
                      AND review_state IN ('approved', 'auto_approved')
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
    normalized = [str(item).strip() for item in parsed if isinstance(item, str) and str(item).strip()]
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


def _passes_auto_approval_gates(
    *,
    decision_status: str,
    decision_score: float,
    top_candidates: list[object],
    mention_entity_type: str,
    min_score: float,
    min_margin: float,
) -> bool:
    """Evaluate whether one pending item is safe to auto-approve."""
    if decision_status != "needs_review":
        return False
    if decision_score < min_score:
        return False
    if not top_candidates:
        return False
    first = top_candidates[0] if isinstance(top_candidates[0], dict) else None
    if first is None:
        return False
    first_type = first.get("entity_type")
    if not isinstance(first_type, str) or first_type != mention_entity_type:
        return False
    first_score = _as_float(first.get("fused_score"), fallback=0.0)
    second_score = 0.0
    if len(top_candidates) > 1 and isinstance(top_candidates[1], dict):
        second_score = _as_float(top_candidates[1].get("fused_score"), fallback=0.0)
    return (first_score - second_score) >= min_margin


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
