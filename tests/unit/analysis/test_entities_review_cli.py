"""Unit tests for entities-review CLI commands."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from ddd_policy_tracer.analysis.entities.models import EntityMention
from ddd_policy_tracer.analysis.entities.review_queue import SQLiteReviewQueueRepository
from ddd_policy_tracer.cli import run_cli


def _seed_review_item(path: Path) -> None:
    """Create one pending review queue item for CLI behavior tests."""
    repository = SQLiteReviewQueueRepository(sqlite_path=path)
    repository.enqueue_resolver_decisions(
        mentions=[
            EntityMention(
                entity_id="entity_1",
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
                    "decision_status": "needs_review",
                    "catalog_eligible": True,
                    "decision_score": 0.7,
                    "reason_codes": ["threshold_review"],
                    "selected_candidate_key": None,
                    "top_candidates": [
                        {
                            "canonical_entity_key": "policy:clean-energy-act",
                            "canonical_name": "Clean Energy Act",
                            "entity_type": "POLICY",
                            "fused_score": 0.7,
                        },
                    ],
                },
            ),
        ],
    )


def _seed_org_review_item(path: Path) -> None:
    """Create one ORG pending item for interactive curation tests."""
    repository = SQLiteReviewQueueRepository(sqlite_path=path)
    repository.enqueue_resolver_decisions(
        mentions=[
            EntityMention(
                entity_id="entity_org_1",
                chunk_id="chunk_org_1",
                source_id="australia_institute",
                source_document_id="https://example.org/report-2",
                document_checksum="checksum-2",
                start_char=4,
                end_char=22,
                mention_text="Australia Institute",
                normalized_mention_text="Australia Institute",
                entity_type="ORG",
                confidence=0.9,
                extractor_version="robust-ensemble-v1",
                metadata={
                    "decision_status": "new_candidate",
                    "catalog_eligible": True,
                    "decision_score": 0.42,
                    "reason_codes": ["threshold_new_candidate"],
                    "selected_candidate_key": None,
                    "top_candidates": [],
                },
            ),
        ],
    )


def test_entities_review_list_returns_items(tmp_path: Path) -> None:
    """List review queue items through top-level CLI route."""
    review_db_path = tmp_path / "review.db"
    _seed_review_item(review_db_path)
    output = StringIO()

    exit_code = run_cli(
        [
            "entities-review",
            "list",
            "--review-db-path",
            str(review_db_path),
            "--state",
            "pending",
        ],
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    assert exit_code == 0
    assert len(payload["items"]) == 1
    assert payload["items"][0]["mention_entity_id"] == "entity_1"


def test_entities_review_decide_updates_review_state(tmp_path: Path) -> None:
    """Apply explicit approve decision through entities-review CLI."""
    review_db_path = tmp_path / "review.db"
    _seed_review_item(review_db_path)
    output = StringIO()

    exit_code = run_cli(
        [
            "entities-review",
            "decide",
            "--review-db-path",
            str(review_db_path),
            "--mention-entity-id",
            "entity_1",
            "--decision",
            "approved",
            "--canonical-entity-key",
            "policy:clean-energy-act",
            "--canonical-name",
            "Clean Energy Act",
            "--entity-type",
            "POLICY",
            "--aliases",
            "Clean Energy Act,CEA",
        ],
        stdout=output,
    )

    payload = json.loads(output.getvalue())
    repository = SQLiteReviewQueueRepository(sqlite_path=review_db_path)
    approved = repository.list_review_items(state="approved")
    assert exit_code == 0
    assert payload["review_state"] == "approved"
    assert len(approved) == 1
    assert approved[0]["canonical_entity_key"] == "policy:clean-energy-act"


def test_entities_review_curate_supports_skip_action(tmp_path: Path) -> None:
    """Interactive curate flow can skip one pending candidate."""
    review_db_path = tmp_path / "review.db"
    chunk_state_path = tmp_path / "chunks.jsonl"
    _seed_org_review_item(review_db_path)
    chunk_state_path.write_text(
        json.dumps(
            {
                "chunk_id": "chunk_org_1",
                "source_id": "australia_institute",
                "source_document_id": "https://example.org/report-2",
                "document_checksum": "checksum-2",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 70,
                "chunk_text": "The Australia Institute released analysis on climate policy.",
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    output = StringIO()
    exit_code = run_cli(
        [
            "entities-review",
            "curate",
            "--review-db-path",
            str(review_db_path),
            "--chunk-state-path",
            str(chunk_state_path),
            "--limit",
            "1",
            "--non-interactive-action",
            "skip",
        ],
        stdout=output,
    )

    lines = [line for line in output.getvalue().splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    repository = SQLiteReviewQueueRepository(sqlite_path=review_db_path)
    rejected = repository.list_review_items(state="rejected")

    assert exit_code == 0
    assert payload["processed_items"] == 1
    assert payload["skipped_items"] == 1
    assert len(rejected) == 1
    assert rejected[0]["mention_entity_id"] == "entity_org_1"


def test_entities_review_curate_prints_non_empty_context(tmp_path: Path) -> None:
    """Curate output includes a sentence-window context snippet."""
    review_db_path = tmp_path / "review.db"
    chunk_state_path = tmp_path / "chunks.jsonl"
    _seed_org_review_item(review_db_path)
    chunk_state_path.write_text(
        json.dumps(
            {
                "chunk_id": "chunk_org_1",
                "source_id": "australia_institute",
                "source_document_id": "https://example.org/report-2",
                "document_checksum": "checksum-2",
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 80,
                "chunk_text": "The Australia Institute released analysis on climate policy.",
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    output = StringIO()
    exit_code = run_cli(
        [
            "entities-review",
            "curate",
            "--review-db-path",
            str(review_db_path),
            "--chunk-state-path",
            str(chunk_state_path),
            "--limit",
            "1",
            "--non-interactive-action",
            "skip",
        ],
        stdout=output,
    )

    content = output.getvalue()
    assert exit_code == 0
    assert "context: " in content
    assert "Australia Institute" in content
