"""Unit tests for entities catalog command routing and import CLI."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from ddd_policy_tracer.cli import run_cli


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
