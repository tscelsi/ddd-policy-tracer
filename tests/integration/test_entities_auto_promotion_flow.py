"""Integration tests for auto-approval and catalog promotion loop."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.entities.catalog import import_seed_catalog, promote_review_items
from ddd_policy_tracer.analysis.entities.models import EntityMention
from ddd_policy_tracer.analysis.entities.review_queue import SQLiteReviewQueueRepository
from ddd_policy_tracer.analysis.entities.run import run


def _write_chunk(path: Path, *, chunk_id: str, text: str) -> None:
    """Write one chunk row for integration loop coverage."""
    row = {
        "chunk_id": chunk_id,
        "source_id": "australia_institute",
        "source_document_id": "https://example.org/report-1",
        "document_checksum": "checksum-1",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": len(text),
        "chunk_text": text,
    }
    path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_unrelated_seed(path: Path) -> None:
    """Write non-matching seed row so target starts outside catalog."""
    row = {
        "canonical_entity_key": "org:unrelated-org",
        "canonical_name": "Unrelated Organization",
        "entity_type": "ORG",
        "aliases": ["UO"],
    }
    path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")


def test_auto_promotion_enables_linked_resolution_on_rerun(tmp_path: Path) -> None:
    """Auto-approve obvious review items then link mention on next run."""
    chunk_state = tmp_path / "chunks.jsonl"
    entities_state = tmp_path / "entities.jsonl"
    seed_path = tmp_path / "seed.jsonl"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    review_db_path = tmp_path / "review.db"

    _write_chunk(
        chunk_state,
        chunk_id="chunk_1",
        text="The Clean Energy Act should reduce emissions.",
    )
    _write_unrelated_seed(seed_path)
    import_seed_catalog(
        seed_path=seed_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
    )

    # Seed one obvious pending item as if surfaced by review queue from prior run.
    review_repository = SQLiteReviewQueueRepository(sqlite_path=review_db_path)
    mention = EntityMention(
        entity_id="entity_manual",
        chunk_id="chunk_1",
        source_id="australia_institute",
        source_document_id="https://example.org/report-1",
        document_checksum="checksum-1",
        start_char=4,
        end_char=20,
        mention_text="Clean Energy Act",
        normalized_mention_text="Clean Energy Act",
        entity_type="POLICY",
        confidence=0.9,
        extractor_version="robust-ensemble-v1",
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
    review_repository.enqueue_resolver_decisions(mentions=[mention])

    promote_review_items(
        review_db_path=review_db_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
        auto_approve=True,
        min_score=0.93,
        min_margin=0.15,
    )

    run(
        chunk_id="chunk_1",
        chunk_state_path=chunk_state,
        entity_state_path=entities_state,
        extractor_kind="robust-ensemble",
        extractor_version="robust-ensemble-v2",
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        review_db_path=review_db_path,
    )

    rows = [
        json.loads(line)
        for line in entities_state.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(
        row.get("metadata", {}).get("decision_status") == "linked"
        and row.get("canonical_entity_key") == "policy:clean-energy-act"
        for row in rows
    )


def test_auto_promotion_leaves_ambiguous_items_for_hitl(tmp_path: Path) -> None:
    """Keep ambiguous pending items unapproved when margin threshold is not met."""
    review_db_path = tmp_path / "review.db"
    repository = SQLiteReviewQueueRepository(sqlite_path=review_db_path)
    repository.enqueue_resolver_decisions(
        mentions=[
            EntityMention(
                entity_id="entity_ambiguous",
                chunk_id="chunk_1",
                source_id="australia_institute",
                source_document_id="https://example.org/report-1",
                document_checksum="checksum-1",
                start_char=4,
                end_char=20,
                mention_text="Clean Energy Act",
                normalized_mention_text="Clean Energy Act",
                entity_type="POLICY",
                confidence=0.9,
                extractor_version="robust-ensemble-v1",
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
                            "fused_score": 0.90,
                        },
                    ],
                },
            ),
        ],
    )

    report = promote_review_items(
        review_db_path=review_db_path,
        catalog_path=tmp_path / "catalog.db",
        vectors_path=tmp_path / "vectors.json",
        catalog_version="catalog-v1",
        auto_approve=True,
        min_score=0.93,
        min_margin=0.15,
    )

    assert report.auto_approved_items == 0
    pending = repository.list_review_items(state="pending")
    assert len(pending) == 1
    assert pending[0]["mention_entity_id"] == "entity_ambiguous"
