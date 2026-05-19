"""Unit tests for automatic database and artifact initialization behavior."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ddd_policy_tracer.analysis.entities.retrieval import HybridCatalogRetriever
from ddd_policy_tracer.analysis.entities.review_queue import SQLiteReviewQueueRepository


def test_hybrid_retriever_auto_initializes_catalog_tables(tmp_path: Path) -> None:
    """Create missing catalog schema on first retrieval call."""
    catalog_path = tmp_path / "entity_catalog.db"
    vectors_path = tmp_path / "entity_vectors.json"
    retriever = HybridCatalogRetriever(catalog_path=catalog_path, vectors_path=vectors_path)

    result = retriever.retrieve(
        mention_text="Clean Energy Act",
        normalized_mention_text="clean energy act",
        mention_entity_type="POLICY",
        top_k=5,
    )

    assert result.candidates == []
    with sqlite3.connect(catalog_path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        ).fetchall()
    table_names = {row[0] for row in tables}
    assert "entity_catalog" in table_names
    assert "catalog_metadata" in table_names


def test_hybrid_retriever_auto_initializes_vectors_file(tmp_path: Path) -> None:
    """Create empty vectors sidecar when vectors artifact is missing."""
    catalog_path = tmp_path / "entity_catalog.db"
    vectors_path = tmp_path / "entity_vectors.json"
    retriever = HybridCatalogRetriever(catalog_path=catalog_path, vectors_path=vectors_path)

    retriever.retrieve(
        mention_text="Australia Institute",
        normalized_mention_text="australia institute",
        mention_entity_type="ORG",
        top_k=5,
    )

    assert vectors_path.exists()
    payload = json.loads(vectors_path.read_text(encoding="utf-8"))
    assert payload["vectors"] == []


def test_review_queue_auto_initializes_review_tables(tmp_path: Path) -> None:
    """Create missing review queue schema on first repository use."""
    review_path = tmp_path / "review.db"
    repository = SQLiteReviewQueueRepository(sqlite_path=review_path)

    items = repository.list_review_items()

    assert items == []
    with sqlite3.connect(review_path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
        ).fetchall()
    table_names = {row[0] for row in tables}
    assert "review_items" in table_names
    assert "review_events" in table_names
