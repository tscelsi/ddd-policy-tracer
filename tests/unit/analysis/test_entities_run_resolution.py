"""Unit tests for entities run integration with deterministic resolver."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.entities.run import run


def _write_chunk(path: Path, *, chunk_id: str, text: str) -> None:
    """Write one chunk fixture record for run-level integration tests."""
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
    """Write one matching catalog seed row for resolver linkage."""
    row = {
        "canonical_entity_key": "policy:clean-energy-act",
        "canonical_name": "Clean Energy Act",
        "entity_type": "POLICY",
        "aliases": ["The Clean Energy Act"],
    }
    path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")


def test_run_applies_resolution_metadata_when_catalog_inputs_are_provided(tmp_path: Path) -> None:
    """Resolve mentions and persist decision metadata in entity output rows."""
    chunk_state = tmp_path / "chunks.jsonl"
    entities_state = tmp_path / "entities.jsonl"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    seed_path = tmp_path / "seed.jsonl"
    _write_chunk(
        chunk_state,
        chunk_id="chunk_1",
        text="The Clean Energy Act should reduce emissions.",
    )
    _write_seed(seed_path)

    from ddd_policy_tracer.analysis.entities import import_seed_catalog

    import_seed_catalog(
        seed_path=seed_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
    )

    report = run(
        chunk_id="chunk_1",
        chunk_state_path=chunk_state,
        entity_state_path=entities_state,
        extractor_kind="robust-ensemble",
        extractor_version="robust-ensemble-v1",
        catalog_path=catalog_path,
        vectors_path=vectors_path,
    )

    assert report.status == "completed"
    rows = [json.loads(line) for line in entities_state.read_text(encoding="utf-8").splitlines() if line]
    assert rows
    first = rows[0]
    assert first["metadata"]["decision_status"] in {
        "linked",
        "needs_review",
        "new_candidate",
        "abstain",
    }
