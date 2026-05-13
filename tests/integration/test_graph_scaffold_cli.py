"""Integration tests for Stage 5 graph scaffold script execution."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL rows for integration graph script tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _claim_row(*, claim_id: str, chunk_id: str, source_id: str) -> dict[str, object]:
    """Build one valid claim row for graph CLI integration tests."""
    return {
        "claim_id": claim_id,
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_document_id": f"https://example.org/{claim_id}",
        "document_checksum": f"checksum-{claim_id}",
        "start_char": 0,
        "end_char": 10,
        "evidence_text": "evidence",
        "normalized_claim_text": claim_id,
        "confidence": 0.9,
        "claim_type": "descriptive",
        "extractor_version": "rules-v1",
        "linked_entities": [],
    }


def _entity_row(
    *,
    entity_id: str,
    chunk_id: str,
    source_id: str,
) -> dict[str, object]:
    """Build one valid entity row for graph CLI integration tests."""
    return {
        "entity_id": entity_id,
        "chunk_id": chunk_id,
        "source_id": source_id,
        "source_document_id": f"https://example.org/{entity_id}",
        "document_checksum": f"checksum-{entity_id}",
        "start_char": 0,
        "end_char": 6,
        "mention_text": "claim",
        "normalized_mention_text": "claim",
        "entity_type": "ORG",
        "confidence": 0.9,
        "extractor_version": "rules-v1",
        "canonical_entity_key": None,
        "canonical_name": None,
    }


def test_graph_scaffold_script_writes_required_artifacts(tmp_path: Path) -> None:
    """Run graph scaffold script and verify required artifact set."""
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

    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise AssertionError("uv executable not found in PATH")

    result = subprocess.run(  # noqa: S603
        [
            uv_executable,
            "run",
            "python",
            "-m",
            "ddd_policy_tracer.analysis.graph.run",
            "--chunks-path",
            str(chunks_path),
            "--claims-path",
            str(claims_path),
            "--entities-path",
            str(entities_path),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    latest_dir = output_root / "latest"
    assert latest_dir.exists()
    for file_name in (
        "graph.json",
        "graph.filtered.json",
        "summary.json",
        "anomalies.json",
        "graph.html",
    ):
        assert (latest_dir / file_name).exists()


def test_graph_scaffold_script_accepts_canonical_opt_in_inputs(tmp_path: Path) -> None:
    """Run graph scaffold CLI with canonical artifacts as explicit inputs."""
    chunks_path = tmp_path / "chunks.jsonl"
    claims_path = tmp_path / "claims_canonical.jsonl"
    entities_path = tmp_path / "entities_canonical.jsonl"
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
                "linked_entities": [
                    {
                        "canonical_entity_key": "entity_canon_abc",
                        "entity_type": "ORG",
                        "canonical_name": "claim",
                        "link_method": "span_overlap",
                        "entity_id": "entity-1",
                    },
                ],
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
                ),
                "canonical_entity_key": "entity_canon_abc",
                "canonical_name": "claim",
            },
        ],
    )

    uv_executable = shutil.which("uv")
    if uv_executable is None:
        raise AssertionError("uv executable not found in PATH")

    result = subprocess.run(  # noqa: S603
        [
            uv_executable,
            "run",
            "python",
            "-m",
            "ddd_policy_tracer.analysis.graph.run",
            "--chunks-path",
            str(chunks_path),
            "--claims-path",
            str(claims_path),
            "--entities-path",
            str(entities_path),
            "--output-root",
            str(output_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    graph_payload = json.loads((output_root / "latest" / "graph.json").read_text(encoding="utf-8"))
    edge_types = {edge["type"] for edge in graph_payload["edges"]}
    assert "MENTIONS" in edge_types
