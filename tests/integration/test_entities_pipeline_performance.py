"""Integration performance checks for robust entities pipeline throughput."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from ddd_policy_tracer.analysis.entities.run import run_bulk
from ddd_policy_tracer.analysis.entities import extractors as entities_extractors


@dataclass
class _StubSentence:
    """Provide sentence objects compatible with extractor expectations."""

    text: str


@dataclass
class _StubDoc:
    """Provide doc objects compatible with spaCy-like extractor API."""

    sents: list[_StubSentence]
    ents: list[object]


class _StubNlp:
    """Provide lightweight nlp parser for performance contract tests."""

    def __call__(self, text: str) -> _StubDoc:
        """Parse one text payload into an empty-entity stub doc."""
        return _StubDoc(sents=[_StubSentence(text=text)], ents=[])

    def pipe(self, texts: list[str]) -> list[_StubDoc]:
        """Parse many text payloads into empty-entity stub docs."""
        return [self(text) for text in texts]


def test_entities_run_bulk_meets_local_throughput_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process 100 chunks under target runtime budget on local hardware."""
    chunks_path = tmp_path / "chunks.jsonl"
    entities_path = tmp_path / "entities.jsonl"
    text = "The Clean Energy Act was discussed by Australia Institute in Queensland."

    with chunks_path.open("w", encoding="utf-8") as handle:
        for index in range(100):
            row = {
                "chunk_id": f"chunk_{index}",
                "source_id": "australia_institute",
                "source_document_id": f"https://example.org/report-{index}",
                "document_checksum": f"checksum-{index}",
                "chunk_index": index,
                "start_char": 0,
                "end_char": len(text),
                "chunk_text": text,
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    monkeypatch.setattr(
        entities_extractors,
        "_build_spacy_fastcoref_nlp",
        lambda _config: _StubNlp(),
    )

    started = time.perf_counter()
    reports = run_bulk(
        chunk_state_path=chunks_path,
        entity_state_path=entities_path,
        extractor_kind="robust-ensemble",
        extractor_version="robust-ensemble-v1",
    )
    elapsed = time.perf_counter() - started

    assert len(reports) == 100
    assert all(report.status == "completed" for report in reports)
    assert elapsed < 2.0
