"""Unit tests for silver dataset QA diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.silver_dataset_qa import run


def _write_jsonl(path: Path, rows: list[str]) -> None:
    """Write provided raw JSONL lines to one fixture file."""
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_run_reports_claims_and_entities_diagnostics(tmp_path: Path) -> None:
    """Compute expected QA counts across valid, invalid, and parse-failure rows."""
    claims_path = tmp_path / "claims_silver.jsonl"
    entities_path = tmp_path / "entities_silver.jsonl"
    output_path = tmp_path / "qa_summary.json"

    _write_jsonl(
        claims_path,
        [
            json.dumps(
                {
                    "chunk_id": "chunk_1",
                    "source_id": "source",
                    "source_document_id": "doc-1",
                    "document_checksum": "checksum-1",
                    "chunk_text": "Text.",
                    "labeling_run_id": "run-1",
                    "labeler_kind": "llm",
                    "labeler_version": "gpt-4.1-mini",
                    "label_prompt_version": "claims-prompt-v1",
                    "dataset_version": "claims-silver-v1",
                    "labeled_at_utc": "2026-05-05T12:34:56Z",
                    "silver_claims": [{"start_char": 0, "end_char": 5}],
                },
                ensure_ascii=True,
            ),
            json.dumps(
                {
                    "chunk_id": "chunk_2",
                    "source_id": "source",
                    "source_document_id": "doc-2",
                    "document_checksum": "checksum-2",
                    "chunk_text": "Other.",
                    "labeling_run_id": "run-1",
                    "labeler_kind": "llm",
                    "labeler_version": "gpt-4.1-mini",
                    "label_prompt_version": "claims-prompt-v1",
                    "dataset_version": "claims-silver-v1",
                    "labeled_at_utc": "2026-05-05T12:34:56Z",
                    "silver_claims": [],
                },
                ensure_ascii=True,
            ),
            "not-json",
        ],
    )

    _write_jsonl(
        entities_path,
        [
            json.dumps(
                {
                    "chunk_id": "chunk_1",
                    "source_id": "source",
                    "source_document_id": "doc-1",
                    "document_checksum": "checksum-1",
                    "chunk_text": "Clean Energy Act in Queensland.",
                    "labeling_run_id": "run-1",
                    "labeler_kind": "llm",
                    "labeler_version": "gpt-4.1-mini",
                    "label_prompt_version": "entities-prompt-v1",
                    "dataset_version": "entities-silver-v1",
                    "labeled_at_utc": "2026-05-05T12:34:56Z",
                    "silver_entities": [
                        {
                            "start_char": 0,
                            "end_char": 16,
                            "mention_text": "Clean Energy Act",
                            "entity_type": "POLICY",
                        },
                    ],
                },
                ensure_ascii=True,
            ),
            json.dumps(
                {
                    "chunk_id": "chunk_2",
                    "source_id": "source",
                    "source_document_id": "doc-2",
                    "document_checksum": "checksum-2",
                    "chunk_text": "Another text",
                    "labeling_run_id": "run-1",
                    "labeler_kind": "llm",
                    "labeler_version": "gpt-4.1-mini",
                    "label_prompt_version": "entities-prompt-v1",
                    "dataset_version": "entities-silver-v1",
                    "labeled_at_utc": "2026-05-05T12:34:56Z",
                    "silver_entities": [
                        {
                            "start_char": 0,
                            "end_char": 100,
                            "mention_text": "Another text",
                            "entity_type": "ORG",
                        },
                    ],
                },
                ensure_ascii=True,
            ),
            json.dumps(
                {
                    "chunk_id": "chunk_3",
                    "source_id": "source",
                    "source_document_id": "doc-3",
                    "document_checksum": "checksum-3",
                    "chunk_text": "Third text",
                    "labeling_run_id": "run-1",
                    "labeler_kind": "llm",
                    "labeler_version": "gpt-4.1-mini",
                    "label_prompt_version": "entities-prompt-v1",
                    "dataset_version": "entities-silver-v1",
                    "labeled_at_utc": "2026-05-05T12:34:56Z",
                    "silver_entities": [],
                },
                ensure_ascii=True,
            ),
            "not-json",
        ],
    )

    summary = run(
        claims_dataset_path=claims_path,
        entities_dataset_path=entities_path,
        output_path=output_path,
    )

    claims = summary["claims"]
    assert claims["parsed_rows"] == 2
    assert claims["parse_failures"] == 1
    assert claims["invalid_rows"] == 0
    assert claims["total_claims"] == 1
    assert claims["empty_label_rows"] == 1
    assert claims["empty_label_rate"] == 0.5

    entities = summary["entities"]
    assert entities["parsed_rows"] == 3
    assert entities["parse_failures"] == 1
    assert entities["invalid_rows"] == 0
    assert entities["total_entities"] == 1
    assert entities["offset_mismatch_mentions"] == 1
    assert entities["empty_label_rows"] == 1
    assert entities["type_distribution"] == {"POLICY": 1}

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["entities"]["total_entities"] == 1
