"""Application service for Stage 5 graph artifact scaffolding."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ddd_policy_tracer.utils.time_helpers import utc_now_isoformat

from .contracts import (
    GRAPH_SCHEMA_VERSION,
    GraphArtifact,
    GraphSummary,
    GraphThresholds,
)
from .materializer import materialize_publisher_claim_graph
from .repositories import JsonlClaimRepository
from .sinks import GraphSink, JsonArtifactSink


@dataclass(frozen=True)
class GraphScaffoldResult:
    """Report run directories and summary for one scaffold execution."""

    run_directory: Path
    latest_directory: Path
    summary: GraphSummary


def scaffold_graph_artifacts(
    *,
    chunks_path: Path,
    claims_path: Path,
    entities_path: Path,
    output_root: Path,
    thresholds: GraphThresholds,
    source_id: str | None = None,
) -> GraphScaffoldResult:
    """Write Stage 5 graph scaffold artifacts for one execution."""
    run_directory = output_root / _build_run_directory_name()
    latest_directory = output_root / "latest"
    sink: GraphSink = JsonArtifactSink(output_dir=run_directory)

    generated_at = utc_now_isoformat()
    claim_repository = JsonlClaimRepository(path=claims_path)
    claims = claim_repository.list_claims(source_id=source_id)
    materialized = materialize_publisher_claim_graph(claims=claims)

    stats = {
        "chunks_input_rows": _count_jsonl_rows(chunks_path),
        "claims_input_rows": _count_jsonl_rows(claims_path),
        "entities_input_rows": _count_jsonl_rows(entities_path),
        "nodes": len(materialized.nodes),
        "edges": len(materialized.edges),
        **materialized.stats,
    }
    artifact = GraphArtifact(
        schema_version=GRAPH_SCHEMA_VERSION,
        generated_at=generated_at,
        thresholds=thresholds,
        stats=stats,
        nodes=materialized.nodes,
        edges=materialized.edges,
    )
    summary = GraphSummary(
        schema_version=GRAPH_SCHEMA_VERSION,
        generated_at=generated_at,
        inputs={
            "chunks_path": str(chunks_path),
            "claims_path": str(claims_path),
            "entities_path": str(entities_path),
            "source_id_filter": source_id or "",
        },
        output_directory=str(run_directory),
        latest_directory=str(latest_directory),
        thresholds=thresholds,
        stats=stats,
    )

    sink.write_full_graph(artifact=artifact)
    sink.write_filtered_graph(artifact=artifact)
    sink.write_summary(summary=summary)
    _write_placeholder_artifacts(output_directory=run_directory, generated_at=generated_at)
    _copy_run_artifacts_to_latest(run_directory=run_directory, latest_directory=latest_directory)

    return GraphScaffoldResult(
        run_directory=run_directory,
        latest_directory=latest_directory,
        summary=summary,
    )


def _build_run_directory_name() -> str:
    """Build one UTC run-folder identifier safe for filesystem paths."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _count_jsonl_rows(path: Path) -> int:
    """Count non-empty JSONL rows for one input artifact path."""
    rows = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows += 1
    return rows


def _copy_run_artifacts_to_latest(*, run_directory: Path, latest_directory: Path) -> None:
    """Replace latest artifacts with files from one run directory."""
    latest_directory.mkdir(parents=True, exist_ok=True)
    for source_file in run_directory.iterdir():
        if not source_file.is_file():
            continue
        shutil.copy2(source_file, latest_directory / source_file.name)


def _write_placeholder_artifacts(*, output_directory: Path, generated_at: str) -> None:
    """Write placeholder anomalies and HTML artifacts for scaffold output."""
    anomalies_path = output_directory / "anomalies.json"
    anomalies_payload = {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "generated_at": generated_at,
        "anomaly_count": 0,
        "anomalies": [],
    }
    anomalies_path.write_text(
        json.dumps(anomalies_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    html_path = output_directory / "graph.html"
    html_path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html lang=\"en\">",
                "  <head>",
                "    <meta charset=\"utf-8\">",
                "    <title>Policy Tracer Graph (Scaffold)</title>",
                "  </head>",
                "  <body>",
                "    <h1>Policy Tracer Graph Scaffold</h1>",
                "    <p>Graph rendering is not implemented in this scaffold run.</p>",
                "  </body>",
                "</html>",
                "",
            ],
        ),
        encoding="utf-8",
    )
