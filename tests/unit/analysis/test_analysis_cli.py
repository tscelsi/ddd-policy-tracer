"""Unit tests for the basic chunking CLI command."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

from ddd_policy_tracer.cli import run_cli


def _write_acquisition_jsonl(path: Path) -> None:
    """Write one filesystem acquisition record for chunking inputs."""
    record = {
        "source_id": "australia_institute",
        "source_document_id": "https://example.org/report-1",
        "source_url": "https://example.org/report-1/",
        "published_at": "2024-01-01T00:00:00+00:00",
        "retrieved_at": "2026-04-30T00:00:00+00:00",
        "checksum": "checksum-1",
        "normalized_text": "abcdefghijklmnopqrstuvwxyz",
        "raw_content_ref": "/tmp/report-1.bin",
        "content_type": "application/pdf",
        "created_at": "2026-04-30T00:00:00+00:00",
        "updated_at": "2026-04-30T00:00:00+00:00",
    }
    path.write_text(
        json.dumps(record, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def test_chunk_cli_persists_chunks_from_filesystem_acquisition_state(
    tmp_path: Path,
) -> None:
    """Chunk one acquired document and persist JSONL chunk output."""
    acquisition_state_path = tmp_path / "acquisition.jsonl"
    chunk_state_path = tmp_path / "chunks.jsonl"
    _write_acquisition_jsonl(acquisition_state_path)
    output = StringIO()

    exit_code = run_cli(
        [
            "chunk",
            "--source",
            "australia_institute",
            "--state-path",
            str(acquisition_state_path),
            "--repository-backend",
            "filesystem",
            "--chunk-state-path",
            str(chunk_state_path),
            "--chunk-repository-backend",
            "filesystem",
            "--chunk-size-chars",
            "10",
            "--chunk-overlap-chars",
            "3",
        ],
        stdout=output,
    )

    rendered = output.getvalue()
    assert exit_code == 0
    assert "source=australia_institute" in rendered
    assert "processed_documents=1" in rendered
    assert "chunked_documents=1" in rendered
    assert "persisted_chunks=4" in rendered
    assert chunk_state_path.exists()
