"""Unit tests for filesystem chunk repository adapter behavior."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.claims import FilesystemChunkRepository


def _write_chunk_record(*, path: Path, chunk_id: str) -> None:
    """Write one chunk JSONL record for repository adapter tests."""
    record = {
        "chunk_id": chunk_id,
        "source_id": "australia_institute",
        "source_document_id": "https://example.org/report-1",
        "document_checksum": "checksum-1",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 42,
        "chunk_text": "Policy should reduce emissions by 4.9%.",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def test_filesystem_chunk_repository_gets_chunk_by_id(tmp_path: Path) -> None:
    """Return chunk model when JSONL contains matching chunk identity."""
    state_path = tmp_path / "chunks.jsonl"
    _write_chunk_record(path=state_path, chunk_id="chunk_1")
    _write_chunk_record(path=state_path, chunk_id="chunk_2")
    repository = FilesystemChunkRepository(state_path)

    chunk = repository.get_chunk(chunk_id="chunk_2")

    assert chunk is not None
    assert chunk.chunk_id == "chunk_2"
    assert chunk.source_id == "australia_institute"
    assert chunk.document_checksum == "checksum-1"


def test_filesystem_chunk_repository_returns_none_when_missing(
    tmp_path: Path,
) -> None:
    """Return none when JSONL has no matching chunk identity."""
    state_path = tmp_path / "chunks.jsonl"
    _write_chunk_record(path=state_path, chunk_id="chunk_1")
    repository = FilesystemChunkRepository(state_path)

    assert repository.get_chunk(chunk_id="chunk_missing") is None
