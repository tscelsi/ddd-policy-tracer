"""Integration tests for Stage 5 graph contract stability guarantees."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.graph.run import run


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one JSONL file from rows for graph integration tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _claim_row(*, claim_id: str, chunk_id: str, source_id: str) -> dict[str, object]:
    """Build one valid claim input row for graph integration tests."""
    return {
        "claim_id": claim_id,
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_document_id": f"https://example.org/{claim_id}",
        "document_checksum": f"checksum-{claim_id}",
        "start_char": 0,
        "end_char": 12,
        "evidence_text": f"normalized {claim_id}",
        "normalized_claim_text": f"normalized {claim_id}",
        "confidence": 0.95,
        "claim_type": "descriptive",
        "extractor_version": "rules-v1",
    }


def _entity_row(*, entity_id: str, chunk_id: str, source_id: str) -> dict[str, object]:
    """Build one valid entity input row for graph integration tests."""
    return {
        "entity_id": entity_id,
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_document_id": f"https://example.org/{entity_id}",
        "document_checksum": f"checksum-{entity_id}",
        "start_char": 0,
        "end_char": 10,
        "mention_text": "normalized",
        "normalized_mention_text": "normalized",
        "entity_type": "ORG",
        "confidence": 0.9,
        "extractor_version": "rules-v1",
        "canonical_entity_key": None,
    }


def test_graph_contract_is_schema_versioned_and_complete(tmp_path: Path) -> None:
    """Write complete schema-versioned artifacts for one successful run."""
    chunks_path = tmp_path / "chunks.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    output_root = tmp_path / "graph_runs"
    _write_jsonl(chunks_path, [{"chunk_id": "chunk-1"}])
    _write_jsonl(
        claims_path,
        [_claim_row(claim_id="claim-1", chunk_id="chunk-1", source_id="australia_institute")],
    )
    _write_jsonl(
        entities_path,
        [_entity_row(entity_id="entity-1", chunk_id="chunk-1", source_id="australia_institute")],
    )

    result = run(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
    )

    graph_path = result.run_directory / "graph.json"
    summary_path = result.run_directory / "summary.json"
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    anomalies_payload = json.loads(
        (result.run_directory / "anomalies.json").read_text(encoding="utf-8"),
    )

    assert graph_payload["schema_version"] == "1.0.0"
    assert isinstance(graph_payload["nodes"], list)
    assert isinstance(graph_payload["edges"], list)
    assert "thresholds" in graph_payload

    assert summary_payload["schema_version"] == "1.0.0"
    assert "max_anomalies" in summary_payload
    assert "anomaly_count" in summary_payload
    assert "exit_code" in summary_payload

    assert anomalies_payload["schema_version"] == "1.0.0"
    assert "categories" in anomalies_payload
    assert isinstance(anomalies_payload["anomalies"], list)


def test_graph_nodes_and_edges_are_deterministic_across_reruns(tmp_path: Path) -> None:
    """Keep node and edge identities stable for unchanged input artifacts."""
    chunks_path = tmp_path / "chunks.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    output_root = tmp_path / "graph_runs"
    _write_jsonl(chunks_path, [{"chunk_id": "chunk-1"}])
    _write_jsonl(
        claims_path,
        [_claim_row(claim_id="claim-1", chunk_id="chunk-1", source_id="australia_institute")],
    )
    _write_jsonl(
        entities_path,
        [_entity_row(entity_id="entity-1", chunk_id="chunk-1", source_id="australia_institute")],
    )

    first_run = run(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
    )
    second_run = run(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
    )

    first_graph = json.loads((first_run.run_directory / "graph.json").read_text(encoding="utf-8"))
    second_graph = json.loads((second_run.run_directory / "graph.json").read_text(encoding="utf-8"))

    assert first_graph["nodes"] == second_graph["nodes"]
    assert first_graph["edges"] == second_graph["edges"]
