"""Unit tests for claim row aggregation script behavior."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.aggregate_claim_rows import run


def test_run_aggregates_duplicate_chunk_rows_and_dedupes_spans(tmp_path: Path) -> None:
    """Merge repeated chunk rows into one row with sorted unique spans."""
    input_path = tmp_path / "claims_rows.jsonl"
    output_path = tmp_path / "claims_aggregated.jsonl"
    rows = [
        {
            "chunk_id": "chunk_1",
            "chunk_text": "Government should reduce emissions quickly.",
            "silver_claims": [{"start_char": 0, "end_char": 10}],
        },
        {
            "chunk_id": "chunk_1",
            "chunk_text": "Government should reduce emissions quickly.",
            "silver_claims": [
                {"start_char": 0, "end_char": 10},
                {"start_char": 11, "end_char": 32},
            ],
        },
    ]
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = run(input_path=input_path, output_path=output_path)

    output_rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary == {"input_rows": 2, "output_rows": 1, "total_spans": 2}
    assert len(output_rows) == 1
    assert output_rows[0]["chunk_id"] == "chunk_1"
    assert output_rows[0]["silver_claims"] == [
        {"start_char": 0, "end_char": 10},
        {"start_char": 11, "end_char": 32},
    ]
