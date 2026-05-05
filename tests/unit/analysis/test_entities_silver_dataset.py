"""Unit tests for entities silver dataset LLM generation flow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ddd_policy_tracer.analysis.entities.evaluation.build_silver_dataset import run


class StubResponse:
    """Represent one deterministic HTTP response payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Initialize a response with static JSON content."""
        self._payload = payload

    def raise_for_status(self) -> None:
        """No-op status check for successful test responses."""

    def json(self) -> dict[str, Any]:
        """Return static response payload for parser assertions."""
        return self._payload


@dataclass
class StubHttpClient:
    """Return queued responses for deterministic dataset tests."""

    responses: list[dict[str, Any]]

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> StubResponse:
        """Pop one response payload for each LLM request call."""
        _ = (url, headers, json)
        if not self.responses:
            raise AssertionError("No stub responses left for request")
        return StubResponse(self.responses.pop(0))


def _write_chunks(path: Path) -> None:
    """Write deterministic chunk JSONL fixture rows."""
    records = [
        {
            "chunk_id": "chunk_1",
            "source_id": "australia_institute",
            "source_document_id": "https://example.org/report-1",
            "document_checksum": "checksum-1",
            "chunk_text": "Clean Energy Act was discussed in Queensland.",
        },
        {
            "chunk_id": "chunk_2",
            "source_id": "australia_institute",
            "source_document_id": "https://example.org/report-2",
            "document_checksum": "checksum-2",
            "chunk_text": "Australia Institute submitted a report.",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set API key expected by LLM-backed dataset generation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_run_writes_schema_valid_entities_silver_records(tmp_path: Path) -> None:
    """Write entities silver rows with strict types and resolved offsets."""
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "entities_silver.jsonl"
    summary_path = tmp_path / "entities_silver_summary.json"
    _write_chunks(chunks_path)

    summary = run(
        chunks_path=chunks_path,
        output_path=output_path,
        sample_size=2,
        seed=42,
        model="gpt-4.1-mini",
        max_entities_per_chunk=8,
        sleep_seconds=0.0,
        label_prompt_version="entities-prompt-v1",
        dataset_version="entities-silver-v1",
        labeling_run_id="entities_run_001",
        labeled_at_utc="2026-05-05T12:34:56Z",
        summary_output_path=summary_path,
        http_client=StubHttpClient(
            responses=[
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"entities": ['
                                    '{"mention_text": "Clean Energy Act", '
                                    '"entity_type": "POLICY"}, '
                                    '{"mention_text": "Queensland", '
                                    '"entity_type": "JURISDICTION"}'
                                    "]}"
                                ),
                            },
                        },
                    ],
                },
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"entities": ['
                                    '{"mention_text": "Australia Institute", '
                                    '"entity_type": "ORG"}'
                                    "]}"
                                ),
                            },
                        },
                    ],
                },
            ],
        ),
    )

    assert summary["sampled_chunks"] == 2
    assert summary["records_written"] == 2
    assert summary["entities_written"] == 3
    assert summary["invalid_type_mentions"] == 0
    assert summary["offset_mismatch_mentions"] == 0

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    all_mentions = [mention for row in rows for mention in row["silver_entities"]]
    assert {mention["entity_type"] for mention in all_mentions} == {
        "POLICY",
        "JURISDICTION",
        "ORG",
    }
    for mention in all_mentions:
        assert mention["end_char"] > mention["start_char"]

    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted_summary["entities_written"] == 3


def test_run_drops_invalid_types_and_offset_mismatches(tmp_path: Path) -> None:
    """Reject non-v1 types and mention strings missing from chunk text."""
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "entities_silver.jsonl"
    _write_chunks(chunks_path)

    summary = run(
        chunks_path=chunks_path,
        output_path=output_path,
        sample_size=1,
        seed=42,
        model="gpt-4.1-mini",
        max_entities_per_chunk=8,
        sleep_seconds=0.0,
        label_prompt_version="entities-prompt-v1",
        dataset_version="entities-silver-v1",
        labeling_run_id="entities_run_001",
        labeled_at_utc="2026-05-05T12:34:56Z",
        summary_output_path=None,
        http_client=StubHttpClient(
            responses=[
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"entities": ['
                                    '{"mention_text": "Missing Mention", '
                                    '"entity_type": "ORG"}, '
                                    '{"mention_text": "Australia Institute", '
                                    '"entity_type": "INVALID"}'
                                    "]}"
                                ),
                            },
                        },
                    ],
                },
            ],
        ),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["silver_entities"] == []
    assert summary["records_written"] == 1
    assert summary["invalid_type_mentions"] == 1
    assert summary["offset_mismatch_mentions"] == 1


def test_run_logs_parse_failure_when_llm_response_is_malformed(tmp_path: Path) -> None:
    """Handle malformed LLM content deterministically without aborting run."""
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "entities_silver.jsonl"
    _write_chunks(chunks_path)

    summary = run(
        chunks_path=chunks_path,
        output_path=output_path,
        sample_size=1,
        seed=42,
        model="gpt-4.1-mini",
        max_entities_per_chunk=8,
        sleep_seconds=0.0,
        label_prompt_version="entities-prompt-v1",
        dataset_version="entities-silver-v1",
        labeling_run_id="entities_run_001",
        labeled_at_utc="2026-05-05T12:34:56Z",
        summary_output_path=None,
        http_client=StubHttpClient(
            responses=[
                {
                    "choices": [
                        {
                            "message": {
                                "content": "not-json",
                            },
                        },
                    ],
                },
            ],
        ),
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["silver_entities"] == []
    assert summary["parse_failures"] == 1
    failures = summary["chunk_failures"]
    assert isinstance(failures, list)
    assert failures[0]["error_type"] == "llm_parse_failed"
