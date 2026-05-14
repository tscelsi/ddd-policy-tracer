"""Unit tests for entities run review queue integration behavior."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.entities import import_seed_catalog
from ddd_policy_tracer.analysis.entities.review_queue import SQLiteReviewQueueRepository
from ddd_policy_tracer.analysis.entities.run import run


def _write_chunk(path: Path, *, chunk_id: str, text: str) -> None:
    """Write one chunk fixture row for run integration tests."""
    record = {
        "chunk_id": chunk_id,
        "source_id": "australia_institute",
        "source_document_id": "https://example.org/report-1",
        "document_checksum": "checksum-1",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": len(text),
        "chunk_text": text,
    }
    path.write_text(json.dumps(record, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_seed(path: Path) -> None:
    """Write one off-target seed row to force unresolved outcome."""
    row = {
        "canonical_entity_key": "org:unrelated-org",
        "canonical_name": "Unrelated Organization",
        "entity_type": "ORG",
        "aliases": ["UO"],
    }
    path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")


def test_run_enqueues_unresolved_mentions_in_review_queue(tmp_path: Path) -> None:
    """Write unresolved resolution outcomes to review queue database."""
    chunk_state = tmp_path / "chunks.jsonl"
    entities_state = tmp_path / "entities.jsonl"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    review_db_path = tmp_path / "review.db"
    seed_path = tmp_path / "seed.jsonl"

    _write_chunk(
        chunk_state,
        chunk_id="chunk_1",
        text="The Clean Energy Act should reduce emissions.",
    )
    _write_seed(seed_path)
    import_seed_catalog(
        seed_path=seed_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
    )

    run(
        chunk_id="chunk_1",
        chunk_state_path=chunk_state,
        entity_state_path=entities_state,
        extractor_kind="robust-ensemble",
        extractor_version="robust-ensemble-v1",
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        review_db_path=review_db_path,
    )

    repository = SQLiteReviewQueueRepository(sqlite_path=review_db_path)
    items = repository.list_review_items()
    assert items
    assert all(item["decision_status"] in {"needs_review", "new_candidate", "abstain"} for item in items)
    assert repository.count_review_events() >= 1
