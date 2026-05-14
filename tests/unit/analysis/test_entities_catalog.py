"""Unit tests for runtime entity catalog import workflows."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ddd_policy_tracer.analysis.entities.catalog import (
    get_catalog_metadata,
    import_seed_catalog,
)


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
