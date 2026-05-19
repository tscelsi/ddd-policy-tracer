"""Unit tests for entities catalog command routing and import CLI."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from ddd_policy_tracer.cli import run_cli
from ddd_policy_tracer.analysis.entities.models import EntityMention
from ddd_policy_tracer.analysis.entities.review_queue import SQLiteReviewQueueRepository


def _write_seed(path: Path) -> None:
    """Write one minimal seed row for CLI catalog import tests."""
    row = {
        "canonical_entity_key": "org:aus-institute",
        "canonical_name": "Australia Institute",
        "entity_type": "ORG",
        "aliases": ["The Australia Institute"],
    }
    path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")


def test_top_level_cli_routes_entities_catalog_import(tmp_path: Path) -> None:
    """Route entities-catalog import through analysis CLI surface."""
    seed_path = tmp_path / "seed.jsonl"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    _write_seed(seed_path)
    output = StringIO()

    exit_code = run_cli(
        [
            "entities-catalog",
            "import",
            "--seed-path",
            str(seed_path),
            "--catalog-path",
            str(catalog_path),
            "--vectors-path",
            str(vectors_path),
            "--catalog-version",
            "catalog-v1",
        ],
        stdout=output,
    )

    assert exit_code == 0
    assert catalog_path.exists()
    assert vectors_path.exists()


def test_top_level_cli_routes_entities_catalog_promote_with_auto_approve(
    tmp_path: Path,
) -> None:
    """Route catalog promotion command and auto-approve obvious candidates."""
    review_db_path = tmp_path / "review.db"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    output = StringIO()

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

    exit_code = run_cli(
        [
            "entities-catalog",
            "promote",
            "--review-db-path",
            str(review_db_path),
            "--catalog-path",
            str(catalog_path),
            "--vectors-path",
            str(vectors_path),
            "--catalog-version",
            "catalog-v1",
            "--auto-approve",
        ],
        stdout=output,
    )

    assert exit_code == 0
    assert catalog_path.exists()
    assert vectors_path.exists()
