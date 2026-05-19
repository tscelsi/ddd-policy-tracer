"""Unit tests for SQLite review queue persistence and events."""

from __future__ import annotations

from dataclasses import replace
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
            "catalog_eligible": True,
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


def test_review_queue_ignores_catalog_ineligible_mentions(tmp_path: Path) -> None:
    """Skip queue insertion when mention is ineligible for catalog curation."""
    repository = SQLiteReviewQueueRepository(sqlite_path=tmp_path / "review.db")
    ineligible = replace(
        _mention(entity_id="entity_1", status="new_candidate"),
        metadata={
            "decision_status": "new_candidate",
            "catalog_eligible": False,
            "decision_score": 0.2,
            "reason_codes": ["catalog_ineligible"],
            "selected_candidate_key": None,
            "top_candidates": [],
        },
    )

    report = repository.enqueue_resolver_decisions(mentions=[ineligible])

    assert report.queued_items == 0
    assert report.appended_events == 0
    assert repository.list_review_items() == []


def test_review_decision_approves_item_and_sets_canonical_fields(tmp_path: Path) -> None:
    """Store explicit approval decision and canonical field overrides."""
    repository = SQLiteReviewQueueRepository(sqlite_path=tmp_path / "review.db")
    mention = _mention(entity_id="entity_1", status="needs_review")
    repository.enqueue_resolver_decisions(mentions=[mention])

    report = repository.decide_item(
        mention_entity_id="entity_1",
        decision="approved",
        canonical_entity_key="policy:clean-energy-act",
        canonical_name="Clean Energy Act",
        entity_type="POLICY",
        aliases=["The Clean Energy Act", "CEA"],
    )

    assert report.review_state == "approved"
    items = repository.list_review_items(state="approved")
    assert len(items) == 1
    assert items[0]["canonical_entity_key"] == "policy:clean-energy-act"
    assert items[0]["canonical_name"] == "Clean Energy Act"
    assert "CEA" in items[0]["canonical_aliases"]


def test_auto_approve_obvious_only_promotes_high_confidence_pending_items(tmp_path: Path) -> None:
    """Automatically approve only pending obvious candidates by gates."""
    repository = SQLiteReviewQueueRepository(sqlite_path=tmp_path / "review.db")
    strong = replace(
        _mention(entity_id="entity_1", status="needs_review"),
        metadata={
            "decision_status": "needs_review",
            "decision_score": 0.95,
            "reason_codes": ["threshold_review"],
            "top_candidates": [
                {
                    "canonical_entity_key": "policy:clean-energy-act",
                    "canonical_name": "Clean Energy Act",
                    "entity_type": "POLICY",
                    "fused_score": 0.95,
                },
                {
                    "canonical_entity_key": "policy:other",
                    "canonical_name": "Other",
                    "entity_type": "POLICY",
                    "fused_score": 0.70,
                },
            ],
        },
    )
    weak = replace(
        _mention(entity_id="entity_2", status="needs_review"),
        metadata={
            "decision_status": "needs_review",
            "decision_score": 0.65,
            "reason_codes": ["threshold_review"],
            "top_candidates": [
                {
                    "canonical_entity_key": "policy:clean-energy-act",
                    "canonical_name": "Clean Energy Act",
                    "entity_type": "POLICY",
                    "fused_score": 0.65,
                },
                {
                    "canonical_entity_key": "policy:other",
                    "canonical_name": "Other",
                    "entity_type": "POLICY",
                    "fused_score": 0.62,
                },
            ],
        },
    )
    repository.enqueue_resolver_decisions(mentions=[strong, weak])

    report = repository.auto_approve_obvious(min_score=0.93, min_margin=0.15)

    assert report.considered_items == 2
    assert report.auto_approved_items == 1
    auto_items = repository.list_review_items(state="auto_approved")
    assert len(auto_items) == 1
    assert auto_items[0]["mention_entity_id"] == "entity_1"
