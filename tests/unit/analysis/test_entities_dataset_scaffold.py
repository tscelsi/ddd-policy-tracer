"""Unit tests for entity evaluation dataset scaffold generation."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.entities.evaluation.build_dataset_scaffold import run


def _write_chunks(path: Path) -> None:
    """Write deterministic chunk JSONL fixtures for scaffold tests."""
    records = [
        {
            "chunk_id": "chunk_1",
            "source_id": "australia_institute",
            "source_document_id": "https://example.org/report-1",
            "document_checksum": "checksum-1",
            "chunk_text": "Chunk one text",
        },
        {
            "chunk_id": "chunk_2",
            "source_id": "australia_institute",
            "source_document_id": "https://example.org/report-2",
            "document_checksum": "checksum-2",
            "chunk_text": "Chunk two text",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def test_build_dataset_scaffold_writes_sampled_records(tmp_path: Path) -> None:
    """Write sampled scaffold records with empty gold entity labels."""
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "entities_dataset.jsonl"
    _write_chunks(chunks_path)

    run(
        chunks_path=chunks_path,
        output_path=output_path,
        sample_size=1,
        seed=42,
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert set(rows[0]) == {
        "chunk_id",
        "source_id",
        "source_document_id",
        "document_checksum",
        "chunk_text",
        "gold_entities",
    }
    assert rows[0]["gold_entities"] == []
