"""Unit tests for Stage 5 graph scaffold run behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddd_policy_tracer.analysis.graph.run import run


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL rows to one path for graph run tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _claim_row(*, claim_id: str, chunk_id: str, source_id: str) -> dict[str, object]:
    """Build one valid claim JSON row for graph run fixture inputs."""
    return {
        "claim_id": claim_id,
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_document_id": f"https://example.org/{claim_id}",
        "document_checksum": f"checksum-{claim_id}",
        "start_char": 0,
        "end_char": 12,
        "evidence_text": "evidence",
        "normalized_claim_text": f"normalized {claim_id}",
        "confidence": 0.95,
        "claim_type": "descriptive",
        "extractor_version": "rules-v1",
    }


def _entity_row(
    *,
    entity_id: str,
    chunk_id: str,
    source_id: str,
    mention_text: str,
    normalized_mention_text: str,
    entity_type: str,
) -> dict[str, object]:
    """Build one valid entity JSON row for graph run fixture inputs."""
    return {
        "entity_id": entity_id,
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_document_id": f"https://example.org/{entity_id}",
        "document_checksum": f"checksum-{entity_id}",
        "start_char": 0,
        "end_char": len(mention_text),
        "mention_text": mention_text,
        "normalized_mention_text": normalized_mention_text,
        "entity_type": entity_type,
        "confidence": 0.9,
        "extractor_version": "rules-v1",
        "canonical_entity_key": None,
    }


def test_run_writes_timestamped_and_latest_artifacts(tmp_path: Path) -> None:
    """Generate scaffold artifacts under run and latest directories."""
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
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                source_id="australia_institute",
                mention_text="Institute",
                normalized_mention_text="institute",
                entity_type="ORG",
            ),
        ],
    )

    result = run(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
    )

    assert result.run_directory.exists()
    assert result.latest_directory.exists()
    for file_name in (
        "graph.json",
        "graph.filtered.json",
        "summary.json",
        "anomalies.json",
        "graph.html",
    ):
        assert (result.run_directory / file_name).exists()
        assert (result.latest_directory / file_name).exists()


def test_run_summary_includes_schema_and_input_paths(tmp_path: Path) -> None:
    """Persist summary metadata with schema version and inputs."""
    chunks_path = tmp_path / "chunks.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    output_root = tmp_path / "graph_runs"
    _write_jsonl(chunks_path, [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}])
    _write_jsonl(
        claims_path,
        [_claim_row(claim_id="claim-1", chunk_id="chunk-1", source_id="australia_institute")],
    )
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                source_id="australia_institute",
                mention_text="Institute",
                normalized_mention_text="institute",
                entity_type="ORG",
            ),
        ],
    )

    result = run(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
    )

    summary_path = result.run_directory / "summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["schema_version"] == "1.0.0"
    assert summary_payload["inputs"]["chunks_path"] == str(chunks_path)
    assert summary_payload["inputs"]["claims_path"] == str(claims_path)
    assert summary_payload["inputs"]["entities_path"] == str(entities_path)
    assert summary_payload["inputs"]["source_id_filter"] == ""
    assert summary_payload["stats"]["chunks_input_rows"] == 2
    assert summary_payload["stats"]["claims_input_rows"] == 1
    assert summary_payload["stats"]["entities_input_rows"] == 1


def test_run_raises_for_missing_required_input(tmp_path: Path) -> None:
    """Reject run execution when one required input file is missing."""
    chunks_path = tmp_path / "chunks.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    output_root = tmp_path / "graph_runs"
    _write_jsonl(chunks_path, [{"chunk_id": "chunk-1"}])
    _write_jsonl(claims_path, [{"claim_id": "claim-1"}])

    with pytest.raises(ValueError, match="entities input path does not exist"):
        run(
            chunks_path=chunks_path,
            claims_path=claims_path,
            entities_path=entities_path,
            output_root=output_root,
        )


def test_run_materializes_publisher_and_claim_graph_nodes(tmp_path: Path) -> None:
    """Build publisher and claim graph artifacts from claim JSONL records."""
    chunks_path = tmp_path / "chunks.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    output_root = tmp_path / "graph_runs"
    _write_jsonl(chunks_path, [{"chunk_id": "chunk-1"}])
    _write_jsonl(
        claims_path,
        [
            {
                **_claim_row(
                    claim_id="claim-1",
                    chunk_id="chunk-1",
                    source_id="australia_institute",
                ),
            },
        ],
    )
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                source_id="australia_institute",
                mention_text="claim",
                normalized_mention_text="claim",
                entity_type="ORG",
            ),
        ],
    )

    result = run(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
    )

    graph_payload = json.loads((result.run_directory / "graph.json").read_text(encoding="utf-8"))
    node_types = {node["type"] for node in graph_payload["nodes"]}
    edge_types = {edge["type"] for edge in graph_payload["edges"]}
    assert "PublisherOrganization" in node_types
    assert "Claim" in node_types
    assert edge_types == {"MENTIONS", "RAISED"}
    assert graph_payload["stats"]["publisher_nodes"] == 1
    assert graph_payload["stats"]["claim_nodes"] == 1
    assert graph_payload["stats"]["raised_edges"] == 1
    assert graph_payload["stats"]["mentioned_entity_nodes"] == 1
    assert graph_payload["stats"]["mentions_edges"] == 1


def test_run_source_filter_limits_materialized_claims(tmp_path: Path) -> None:
    """Restrict materialized graph results when one source filter is provided."""
    chunks_path = tmp_path / "chunks.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    output_root = tmp_path / "graph_runs"
    _write_jsonl(chunks_path, [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}])
    _write_jsonl(
        claims_path,
        [
            {
                **_claim_row(
                    claim_id="claim-1",
                    chunk_id="chunk-1",
                    source_id="australia_institute",
                ),
            },
            {
                **_claim_row(
                    claim_id="claim-2",
                    chunk_id="chunk-2",
                    source_id="lowy_institute",
                ),
            },
        ],
    )
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                source_id="australia_institute",
                mention_text="claim",
                normalized_mention_text="claim",
                entity_type="ORG",
            ),
            _entity_row(
                entity_id="entity-2",
                chunk_id="chunk-2",
                source_id="lowy_institute",
                mention_text="claim",
                normalized_mention_text="claim",
                entity_type="ORG",
            ),
        ],
    )

    result = run(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
        source_id="australia_institute",
    )

    summary_path = result.run_directory / "summary.json"
    graph_path = result.run_directory / "graph.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert summary_payload["inputs"]["source_id_filter"] == "australia_institute"
    claim_nodes = [node for node in graph_payload["nodes"] if node["type"] == "Claim"]
    entity_nodes = [node for node in graph_payload["nodes"] if node["type"] == "MentionedEntity"]
    assert len(claim_nodes) == 1
    assert len(entity_nodes) == 1
    assert claim_nodes[0]["properties"]["source_id"] == "australia_institute"


def test_run_uses_type_sensitive_entity_dedup_key(tmp_path: Path) -> None:
    """Keep distinct entity nodes when same text appears under different types."""
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
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                source_id="australia_institute",
                mention_text="claim",
                normalized_mention_text="claim",
                entity_type="ORG",
            ),
            _entity_row(
                entity_id="entity-2",
                chunk_id="chunk-1",
                source_id="australia_institute",
                mention_text="claim",
                normalized_mention_text="claim",
                entity_type="PROGRAM",
            ),
        ],
    )

    result = run(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
    )

    graph_payload = json.loads((result.run_directory / "graph.json").read_text(encoding="utf-8"))
    entity_nodes = [node for node in graph_payload["nodes"] if node["type"] == "MentionedEntity"]
    assert len(entity_nodes) == 2


def test_run_writes_filtered_connected_triads_with_default_thresholds(tmp_path: Path) -> None:
    """Keep only connected triads that satisfy default confidence thresholds."""
    chunks_path = tmp_path / "chunks.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    output_root = tmp_path / "graph_runs"
    _write_jsonl(chunks_path, [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}])
    _write_jsonl(
        claims_path,
        [
            {
                **_claim_row(
                    claim_id="claim-1",
                    chunk_id="chunk-1",
                    source_id="australia_institute",
                ),
                "confidence": 0.95,
            },
            {
                **_claim_row(
                    claim_id="claim-2",
                    chunk_id="chunk-2",
                    source_id="australia_institute",
                ),
                "confidence": 0.5,
            },
        ],
    )
    _write_jsonl(
        entities_path,
        [
            {
                **_entity_row(
                    entity_id="entity-1",
                    chunk_id="chunk-1",
                    source_id="australia_institute",
                    mention_text="normalized",
                    normalized_mention_text="normalized",
                    entity_type="ORG",
                ),
                "confidence": 0.8,
            },
            {
                **_entity_row(
                    entity_id="entity-2",
                    chunk_id="chunk-2",
                    source_id="australia_institute",
                    mention_text="normalized",
                    normalized_mention_text="normalized",
                    entity_type="ORG",
                ),
                "confidence": 0.8,
            },
        ],
    )

    result = run(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
    )

    full_payload = json.loads((result.run_directory / "graph.json").read_text(encoding="utf-8"))
    filtered_payload = json.loads(
        (result.run_directory / "graph.filtered.json").read_text(encoding="utf-8"),
    )
    assert len(full_payload["nodes"]) > len(filtered_payload["nodes"])
    assert len(full_payload["edges"]) > len(filtered_payload["edges"])
    filtered_claim_ids = {
        node["id"]
        for node in filtered_payload["nodes"]
        if node["type"] == "Claim"
    }
    assert filtered_claim_ids == {"claim-1"}
    assert filtered_payload["thresholds"]["claim_confidence_min"] == 0.6
    assert filtered_payload["thresholds"]["entity_confidence_min"] == 0.7


def test_run_applies_override_thresholds_to_filtered_graph(tmp_path: Path) -> None:
    """Allow lower threshold overrides to include additional connected claims."""
    chunks_path = tmp_path / "chunks.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    output_root = tmp_path / "graph_runs"
    _write_jsonl(chunks_path, [{"chunk_id": "chunk-1"}, {"chunk_id": "chunk-2"}])
    _write_jsonl(
        claims_path,
        [
            {
                **_claim_row(
                    claim_id="claim-1",
                    chunk_id="chunk-1",
                    source_id="australia_institute",
                ),
                "confidence": 0.95,
            },
            {
                **_claim_row(
                    claim_id="claim-2",
                    chunk_id="chunk-2",
                    source_id="australia_institute",
                ),
                "confidence": 0.5,
            },
        ],
    )
    _write_jsonl(
        entities_path,
        [
            _entity_row(
                entity_id="entity-1",
                chunk_id="chunk-1",
                source_id="australia_institute",
                mention_text="normalized",
                normalized_mention_text="normalized",
                entity_type="ORG",
            ),
            _entity_row(
                entity_id="entity-2",
                chunk_id="chunk-2",
                source_id="australia_institute",
                mention_text="normalized",
                normalized_mention_text="normalized",
                entity_type="ORG",
            ),
        ],
    )

    result = run(
        chunks_path=chunks_path,
        claims_path=claims_path,
        entities_path=entities_path,
        output_root=output_root,
        claim_confidence_min=0.4,
        entity_confidence_min=0.7,
    )

    filtered_payload = json.loads(
        (result.run_directory / "graph.filtered.json").read_text(encoding="utf-8"),
    )
    filtered_claim_ids = {
        node["id"]
        for node in filtered_payload["nodes"]
        if node["type"] == "Claim"
    }
    assert filtered_claim_ids == {"claim-1", "claim-2"}
