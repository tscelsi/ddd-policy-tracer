"""Unit tests for entities evaluation tooling behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ddd_policy_tracer.analysis.entities.evaluation.evaluate_extractor import (
    _to_match_key,
    run,
)


def _write_dataset(path: Path, records: list[dict[str, object]]) -> None:
    """Write JSONL evaluation records for deterministic evaluator tests."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def test_to_match_key_rejects_invalid_entity_type() -> None:
    """Reject non-v1 entity types in strict evaluator matching logic."""
    with pytest.raises(ValueError, match="entity_type must be one of strict v1 values"):
        _to_match_key(
            chunk_id="chunk_1",
            start_char=0,
            end_char=10,
            entity_type="INVALID",
        )


def test_entities_evaluator_reports_micro_and_per_type_metrics(
    tmp_path: Path,
) -> None:
    """Compute metrics and gate outcomes for one deterministic fixture set."""
    dataset_path = tmp_path / "entities_eval.jsonl"
    _write_dataset(
        dataset_path,
        [
            {
                "chunk_id": "chunk_1",
                "source_id": "australia_institute",
                "source_document_id": "https://example.org/report-1",
                "document_checksum": "checksum-1",
                "chunk_text": (
                    "Clean Energy Act was discussed by Australia Institute in Queensland."
                ),
                "gold_entities": [
                    {
                        "start_char": 0,
                        "end_char": 16,
                        "entity_type": "POLICY",
                        "mention_text": "Clean Energy Act",
                    },
                    {
                        "start_char": 34,
                        "end_char": 53,
                        "entity_type": "ORG",
                        "mention_text": "Australia Institute",
                    },
                ],
            },
        ],
    )

    summary = run(
        dataset_path=dataset_path,
        output_path=None,
        policy_threshold=1.0,
        org_threshold=1.0,
        person_threshold=1.0,
        jurisdiction_threshold=1.0,
        program_threshold=1.0,
    )

    assert summary["dataset_records"] == 1
    assert summary["extractor"] == "rule"
    assert summary["total_tp"] == 2
    assert summary["total_fp"] == 1
    assert summary["total_fn"] == 0
    assert summary["micro_precision"] == 2 / 3
    assert summary["micro_recall"] == 1.0
    assert summary["per_type"]["POLICY"]["tp"] == 1
    assert summary["per_type"]["ORG"]["tp"] == 1
    assert summary["gates"]["micro_recall_gte_0_30"] is True
    assert summary["all_gates_passed"] is False


def test_entities_evaluator_writes_output_report_when_path_provided(
    tmp_path: Path,
) -> None:
    """Write evaluator output JSON when caller requests report path."""
    dataset_path = tmp_path / "entities_eval.jsonl"
    output_path = tmp_path / "evaluation_report.json"
    _write_dataset(
        dataset_path,
        [
            {
                "chunk_id": "chunk_1",
                "source_id": "australia_institute",
                "source_document_id": "https://example.org/report-1",
                "document_checksum": "checksum-1",
                "chunk_text": "Clean Energy Act.",
                "gold_entities": [
                    {
                        "start_char": 0,
                        "end_char": 16,
                        "entity_type": "POLICY",
                        "mention_text": "Clean Energy Act",
                    },
                ],
            },
        ],
    )

    summary = run(
        dataset_path=dataset_path,
        output_path=output_path,
        policy_threshold=1.0,
        org_threshold=1.0,
        person_threshold=1.0,
        jurisdiction_threshold=1.0,
        program_threshold=1.0,
    )

    assert output_path.exists()
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["total_tp"] == summary["total_tp"]
