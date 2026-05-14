"""Integration tests for end-to-end entities to graph pipeline contracts."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.canonicalization.run import run_entities
from ddd_policy_tracer.analysis.entities import import_seed_catalog
from ddd_policy_tracer.analysis.entities.run import run
from ddd_policy_tracer.analysis.graph.run import run as run_graph


def _write_chunk(path: Path, *, chunk_id: str, text: str) -> None:
    """Write one chunk row for pipeline integration coverage."""
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
    """Write one seed row for matching canonical catalog lookup."""
    row = {
        "canonical_entity_key": "policy:clean-energy-act",
        "canonical_name": "Clean Energy Act",
        "entity_type": "POLICY",
        "aliases": ["The Clean Energy Act"],
    }
    path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")


def _write_claim(path: Path, *, chunk_id: str) -> None:
    """Write one claim row linked to chunk for graph compatibility checks."""
    row = {
        "claim_id": "claim_1",
        "chunk_id": chunk_id,
        "source_id": "australia_institute",
        "source_document_id": "https://example.org/report-1",
        "document_checksum": "checksum-1",
        "start_char": 0,
        "end_char": 44,
        "evidence_text": "The Clean Energy Act should reduce emissions.",
        "normalized_claim_text": "clean energy act should reduce emissions",
        "confidence": 0.9,
        "claim_type": "descriptive",
        "extractor_version": "rules-v1",
    }
    path.write_text(json.dumps(row, ensure_ascii=True) + "\n", encoding="utf-8")


def test_entities_to_canonicalization_to_graph_pipeline_compatibility(tmp_path: Path) -> None:
    """Validate robust entities output remains compatible with graph materialization."""
    chunks_path = tmp_path / "chunks.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    entities_canonical_path = tmp_path / "entities_canonical.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    graph_root = tmp_path / "graph_runs"
    seed_path = tmp_path / "seed.jsonl"
    catalog_path = tmp_path / "catalog.db"
    vectors_path = tmp_path / "vectors.json"
    _write_chunk(chunks_path, chunk_id="chunk_1", text="The Clean Energy Act should reduce emissions.")
    _write_seed(seed_path)
    _write_claim(claims_path, chunk_id="chunk_1")

    import_seed_catalog(
        seed_path=seed_path,
        catalog_path=catalog_path,
        vectors_path=vectors_path,
        catalog_version="catalog-v1",
    )
    run(
        chunk_id="chunk_1",
        chunk_state_path=chunks_path,
        entity_state_path=entities_path,
        extractor_kind="robust-ensemble",
        extractor_version="robust-ensemble-v1",
        catalog_path=catalog_path,
        vectors_path=vectors_path,
    )
    run_entities(
        entities_path=entities_path,
        entities_canonical_path=entities_canonical_path,
        entity_canonicalizer_version="entity-v1",
    )

    result = run_graph(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_canonical_path,
        output_root=graph_root,
    )

    assert result.exit_code == 0
    graph_payload = json.loads((result.run_directory / "graph.json").read_text(encoding="utf-8"))
    assert graph_payload["stats"]["claim_nodes"] == 1
    assert graph_payload["stats"]["mentioned_entity_nodes"] >= 1
