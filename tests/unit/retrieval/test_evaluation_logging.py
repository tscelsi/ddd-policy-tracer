"""Tests for persisted retrieval evaluation results."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.retrieval.evaluation import (
    CaseEvaluation,
    EvaluationReport,
    RetrievedChunkRecord,
    append_evaluation_log,
)


def test_evaluation_log_includes_retrieved_chunks_and_errors(tmp_path: Path) -> None:
    """Persist returned passage text and per-case errors in JSON-lines format."""
    report = EvaluationReport(
        cases=(
            CaseEvaluation(
                case_id="case-one",
                passed=False,
                target_ranks={"chunk-a": None},
                stale_chunk_ids=(),
                retrieved_chunks=(
                    RetrievedChunkRecord(
                        rank=1,
                        chunk_id="chunk-b",
                        chunk_text="The returned passage.",
                        source_document_id="speech-one",
                        metadata='{"date":"2024-01-01"}',
                    ),
                ),
                error="UnsupportedFilterError: Unsupported date constraint",
            ),
        ),
        unanswerable_case_ids=(),
    )
    log_path = tmp_path / "retrieval.log"

    append_evaluation_log(report, log_path)

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "case_id": "case-one",
            "passed": False,
            "target_ranks": {"chunk-a": None},
            "stale_chunk_ids": [],
            "error": "UnsupportedFilterError: Unsupported date constraint",
            "retrieved_chunks": [
                {
                    "rank": 1,
                    "chunk_id": "chunk-b",
                    "chunk_text": "The returned passage.",
                    "source_document_id": "speech-one",
                    "metadata": '{"date":"2024-01-01"}',
                },
            ],
        },
    ]
