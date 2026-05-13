"""Unit tests for canonicalization wiring via top-level analysis CLI."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from ddd_policy_tracer.cli import run_cli


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL fixture rows to one artifact path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def test_top_level_cli_routes_canonicalize_entities(tmp_path: Path) -> None:
    """Route canonicalize entities command through top-level CLI."""
    entities_path = tmp_path / "entities.jsonl"
    entities_canonical_path = tmp_path / "entities_canonical.jsonl"
    _write_jsonl(
        entities_path,
        [
            {
                "entity_id": "entity-1",
                "chunk_id": "chunk-1",
                "source_id": "australia_institute",
                "source_document_id": "https://example.org/report-1",
                "document_checksum": "checksum-1",
                "start_char": 0,
                "end_char": 8,
                "mention_text": "Institute",
                "normalized_mention_text": "Institute",
                "entity_type": "ORG",
                "confidence": 0.9,
                "extractor_version": "rules-v1",
                "canonical_entity_key": None,
                "metadata": None,
            },
        ],
    )

    exit_code = run_cli(
        [
            "canonicalize",
            "entities",
            "--entities-path",
            str(entities_path),
            "--entities-canonical-path",
            str(entities_canonical_path),
            "--entity-canonicalizer-version",
            "entity-v1",
        ],
        stdout=StringIO(),
    )

    assert exit_code == 0
    assert entities_canonical_path.exists()
