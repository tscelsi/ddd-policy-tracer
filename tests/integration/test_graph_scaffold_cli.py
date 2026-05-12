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


def test_graph_scaffold_script_writes_required_artifacts(tmp_path: Path) -> None:
    """Run graph scaffold script and verify required artifact set."""
    chunks_path = tmp_path / "chunks.jsonl"
    claims_path = tmp_path / "claims.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    output_root = tmp_path / "graph_runs"
    _write_jsonl(chunks_path, [{"chunk_id": "chunk-1"}])
    _write_jsonl(claims_path, [{"claim_id": "claim-1"}])
    _write_jsonl(entities_path, [{"entity_id": "entity-1"}])

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
