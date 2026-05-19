"""Unit tests for runtime entity catalog import workflows."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ddd_policy_tracer.analysis.entities.catalog import (
    get_catalog_metadata,
    import_seed_catalog,
    promote_review_items,
)
from ddd_policy_tracer.analysis.entities.models import EntityMention
from ddd_policy_tracer.analysis.entities.review_queue import SQLiteReviewQueueRepository


def _write_seed(path: Path) -> None:
    """Write representative seed rows for catalog import testing."""
    rows = [
        {
            "canonical_entity_key": "org:aus-institute",
            "canonical_name": "Australia Institute",
            "entity_type": "ORG",
            "aliases": ["The Australia Institute", "Australia Institute"],
        },
        {
            "canonical_entity_key": "policy:clean-energy-act",
            "canonical_name": "Clean Energy Act",
            "entity_type": "POLICY",
            "aliases": ["CEA"],
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_import_seed_catalog_inserts_rows_and_writes_vectors(tmp_path: Path) -> None:
    """Import seed rows into SQLite runtime catalog and sidecar vectors."""
    seed_path = tmp_path / "seed.jsonl"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    _write_seed(seed_path)

    report = import_seed_catalog(
        seed_path=seed_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
    )

    assert report.processed_records == 2
    assert report.inserted_records == 2
    assert report.vectors_written == 2
    assert report.catalog_version == "catalog-v1"
    assert len(report.seed_hash) == 64

    with sqlite3.connect(catalog_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM entity_catalog").fetchone()[0]
    assert count == 2

    vectors = json.loads(vectors_path.read_text(encoding="utf-8"))
    assert vectors["catalog_version"] == "catalog-v1"
    assert vectors["seed_hash"] == report.seed_hash
    assert len(vectors["vectors"]) == 2


def test_import_seed_catalog_is_idempotent_for_same_seed(tmp_path: Path) -> None:
    """Re-import unchanged seed without duplicating runtime catalog rows."""
    seed_path = tmp_path / "seed.jsonl"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    _write_seed(seed_path)

    first = import_seed_catalog(
        seed_path=seed_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
    )
    second = import_seed_catalog(
        seed_path=seed_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
    )

    assert first.inserted_records == 2
    assert second.inserted_records == 0
    assert first.seed_hash == second.seed_hash

    with sqlite3.connect(catalog_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM entity_catalog").fetchone()[0]
    assert count == 2


def test_get_catalog_metadata_returns_version_and_seed_hash(tmp_path: Path) -> None:
    """Expose metadata required for compatibility checks."""
    seed_path = tmp_path / "seed.jsonl"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    _write_seed(seed_path)

    report = import_seed_catalog(
        seed_path=seed_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v2",
    )
    metadata = get_catalog_metadata(catalog_path=catalog_path)

    assert metadata == {
        "catalog_version": "catalog-v2",
        "seed_hash": report.seed_hash,
    }


def test_promote_review_items_upserts_catalog_and_rewrites_vectors(tmp_path: Path) -> None:
    """Promote approved review item into catalog and refresh vectors artifact."""
    review_db_path = tmp_path / "review.db"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"

    repository = SQLiteReviewQueueRepository(sqlite_path=review_db_path)
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
                            "fused_score": 0.60,
                        },
                    ],
                },
            ),
        ],
    )

    repository.decide_item(
        mention_entity_id="entity_1",
        decision="approved",
        canonical_entity_key="policy:clean-energy-act",
        canonical_name="Clean Energy Act",
        entity_type="POLICY",
        aliases=["Clean Energy Act"],
    )

    report = promote_review_items(
        review_db_path=review_db_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
        auto_approve=True,
        min_score=0.93,
        min_margin=0.15,
    )

    assert report.auto_approved_items == 0
    assert report.promoted_items == 1
    assert report.inserted_records == 1
    assert report.vectors_written == 1

    with sqlite3.connect(catalog_path) as connection:
        row = connection.execute(
            "SELECT canonical_entity_key, canonical_name FROM entity_catalog",
        ).fetchone()
    assert row == ("policy:clean-energy-act", "Clean Energy Act")

    payload = json.loads(vectors_path.read_text(encoding="utf-8"))
    assert payload["catalog_version"] == "catalog-v1"
    assert len(payload["vectors"]) == 1


def test_promote_review_items_is_idempotent_for_already_promoted_items(tmp_path: Path) -> None:
    """Skip duplicate catalog promotion when item is already promoted."""
    review_db_path = tmp_path / "review.db"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"

    repository = SQLiteReviewQueueRepository(sqlite_path=review_db_path)
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
                            "fused_score": 0.60,
                        },
                    ],
                },
            ),
        ],
    )

    repository.decide_item(
        mention_entity_id="entity_1",
        decision="approved",
        canonical_entity_key="policy:clean-energy-act",
        canonical_name="Clean Energy Act",
        entity_type="POLICY",
        aliases=["Clean Energy Act"],
    )

    first = promote_review_items(
        review_db_path=review_db_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
        auto_approve=True,
        min_score=0.93,
        min_margin=0.15,
    )
    second = promote_review_items(
        review_db_path=review_db_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
        auto_approve=True,
        min_score=0.93,
        min_margin=0.15,
    )

    assert first.promoted_items == 1
    assert second.promoted_items == 0
    with sqlite3.connect(catalog_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM entity_catalog").fetchone()[0]
    assert count == 1
