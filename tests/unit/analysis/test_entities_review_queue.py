"""Unit tests for SQLite review queue persistence and events."""

from __future__ import annotations

from pathlib import Path

from ddd_policy_tracer.analysis.entities import EntityMention
from ddd_policy_tracer.analysis.entities.review_queue import SQLiteReviewQueueRepository


def _mention(*, entity_id: str, status: str) -> EntityMention:
    """Build one mention fixture with resolver decision metadata."""
    return EntityMention(
        entity_id=entity_id,
        chunk_id="chunk_1",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        start_char=0,
        end_char=16,
        mention_text="Clean Energy Act",
        normalized_mention_text="Clean Energy Act",
        entity_type="POLICY",
        confidence=0.9,
        extractor_version="robust-ensemble-v1",
        metadata={
            "decision_status": status,
            "decision_score": 0.7,
            "reason_codes": ["threshold_review"],
            "selected_candidate_key": None,
            "top_candidates": [
                {
                    "canonical_entity_key": "policy:clean-energy-act",
                    "fused_score": 0.7,
                },
            ],
        },
    )


def test_review_queue_persists_unresolved_items_and_events(tmp_path: Path) -> None:
    """Persist unresolved mentions into review_items and review_events tables."""
    repository = SQLiteReviewQueueRepository(sqlite_path=tmp_path / "review.db")
    mentions = [
        _mention(entity_id="entity_1", status="needs_review"),
        _mention(entity_id="entity_2", status="new_candidate"),
        _mention(entity_id="entity_3", status="abstain"),
    ]

    report = repository.enqueue_resolver_decisions(mentions=mentions)

    assert report.queued_items == 3
    assert report.appended_events == 3
    items = repository.list_review_items()
    assert [item["mention_entity_id"] for item in items] == ["entity_1", "entity_2", "entity_3"]
    assert repository.count_review_events() == 3


def test_review_queue_appends_events_replay_safely(tmp_path: Path) -> None:
    """Avoid duplicate review events for identical decision payloads."""
    repository = SQLiteReviewQueueRepository(sqlite_path=tmp_path / "review.db")
    mention = _mention(entity_id="entity_1", status="needs_review")

    first = repository.enqueue_resolver_decisions(mentions=[mention])
    second = repository.enqueue_resolver_decisions(mentions=[mention])

    assert first.appended_events == 1
    assert second.appended_events == 0
    assert repository.count_review_events() == 1


def test_review_queue_ignores_linked_mentions(tmp_path: Path) -> None:
    """Skip queue insertion for already linked mentions."""
    repository = SQLiteReviewQueueRepository(sqlite_path=tmp_path / "review.db")
    linked = _mention(entity_id="entity_1", status="linked")

    report = repository.enqueue_resolver_decisions(mentions=[linked])

    assert report.queued_items == 0
    assert report.appended_events == 0
    assert repository.list_review_items() == []
