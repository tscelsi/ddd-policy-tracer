"""Unit tests for claims silver dataset LLM generation flow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ddd_policy_tracer.analysis.claims.evaluation.build_silver_dataset import run


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
            "chunk_text": "Government should ban new coal projects.",
        },
        {
            "chunk_id": "chunk_2",
            "source_id": "australia_institute",
            "source_document_id": "https://example.org/report-2",
            "document_checksum": "checksum-2",
            "chunk_text": "Policy should reduce emissions by 4.9%.",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set API key expected by LLM-backed dataset generation."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_run_writes_schema_valid_claims_silver_records(tmp_path: Path) -> None:
    """Write claims silver rows with lineage metadata and diagnostics summary."""
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "claims_silver.jsonl"
    summary_path = tmp_path / "claims_silver_summary.json"
    _write_chunks(chunks_path)

    summary = run(
        chunks_path=chunks_path,
        output_path=output_path,
        sample_size=2,
        seed=42,
        model="gpt-4.1-mini",
        max_claims_per_chunk=5,
        sleep_seconds=0.0,
        label_prompt_version="claims-prompt-v1",
        dataset_version="claims-silver-v1",
        labeling_run_id="claims_run_001",
        labeled_at_utc="2026-05-05T12:34:56Z",
        summary_output_path=summary_path,
        http_client=StubHttpClient(
            responses=[
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"claims": ["Government should ban new coal projects."]}'
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
                                    '{"claims": ["Policy should reduce emissions by 4.9%."]}'
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
    assert summary["claims_written"] == 2
    assert summary["parse_failures"] == 0
    assert summary["invalid_rows"] == 0

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["labeling_run_id"] == "claims_run_001"
        assert row["labeler_kind"] == "llm"
        assert row["labeler_version"] == "gpt-4.1-mini"
        assert row["label_prompt_version"] == "claims-prompt-v1"
        assert row["dataset_version"] == "claims-silver-v1"
        assert isinstance(row["silver_claims"], list)
        assert len(row["silver_claims"]) <= 1

    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted_summary["records_written"] == 2


def test_run_logs_parse_failures_and_writes_empty_claims_record(tmp_path: Path) -> None:
    """Handle malformed LLM content deterministically without aborting run."""
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "claims_silver.jsonl"
    _write_chunks(chunks_path)

    summary = run(
        chunks_path=chunks_path,
        output_path=output_path,
        sample_size=1,
        seed=42,
        model="gpt-4.1-mini",
        max_claims_per_chunk=5,
        sleep_seconds=0.0,
        label_prompt_version="claims-prompt-v1",
        dataset_version="claims-silver-v1",
        labeling_run_id="claims_run_001",
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
    assert rows[0]["silver_claims"] == []
    assert summary["records_written"] == 1
    assert summary["parse_failures"] == 1
    failures = summary["chunk_failures"]
    assert isinstance(failures, list)
    assert failures[0]["error_type"] == "llm_parse_failed"


def test_run_writes_multiple_rows_when_one_chunk_has_multiple_claims(tmp_path: Path) -> None:
    """Emit claim-per-row records when one chunk yields multiple claim spans."""
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "claims_silver.jsonl"
    _write_chunks(chunks_path)

    summary = run(
        chunks_path=chunks_path,
        output_path=output_path,
        sample_size=1,
        seed=42,
        model="gpt-4.1-mini",
        max_claims_per_chunk=5,
        sleep_seconds=0.0,
        label_prompt_version="claims-prompt-v1",
        dataset_version="claims-silver-v1",
        labeling_run_id="claims_run_001",
        labeled_at_utc="2026-05-05T12:34:56Z",
        summary_output_path=None,
        http_client=StubHttpClient(
            responses=[
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"claims": ['
                                    '"Government should ban new coal projects.", '
                                    '"coal projects"'
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
    assert len(rows) == 2
    assert all(len(row["silver_claims"]) == 1 for row in rows)
    assert summary["records_written"] == 2
    assert summary["claims_written"] == 2
