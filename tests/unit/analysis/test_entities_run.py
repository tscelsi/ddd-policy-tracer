"""Unit tests for entities run module wiring behavior."""

from __future__ import annotations

import json
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from ddd_policy_tracer.analysis.entities import RuleBasedSentenceEntityExtractor
from ddd_policy_tracer.analysis.entities.run import _build_parser, run


def _write_chunk_record(*, path: Path, chunk_id: str, chunk_text: str) -> None:
    """Write one chunk JSONL record for entities run wiring tests."""
    record = {
        "chunk_id": chunk_id,
        "source_id": "australia_institute",
        "source_document_id": "https://example.org/report-1",
        "document_checksum": "checksum-1",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": len(chunk_text),
        "chunk_text": chunk_text,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def test_run_executes_entities_service_wiring_for_one_chunk(tmp_path: Path) -> None:
    """Run concrete entities extraction wiring for one chunk identifier."""
    chunk_state_path = tmp_path / "chunks.jsonl"
    entity_state_path = tmp_path / "entities.jsonl"
    _write_chunk_record(
        path=chunk_state_path,
        chunk_id="chunk_1",
        chunk_text="The Clean Energy Act was discussed by Australia Institute in Queensland.",
    )

    report = run(
        chunk_id="chunk_1",
        chunk_state_path=chunk_state_path,
        entity_state_path=entity_state_path,
        extractor_version="rules-v1",
    )

    assert report.chunk_id == "chunk_1"
    assert report.status == "completed"
    assert report.entities_extracted >= 1
    assert report.processed_sentences == 1
    persisted_lines = entity_state_path.read_text(encoding="utf-8").splitlines()
    assert len(persisted_lines) == report.entities_extracted


def test_build_parser_defaults_entity_extractor_version_from_env(
    monkeypatch: MonkeyPatch,
) -> None:
    """Default extractor version should come from environment when set."""
    monkeypatch.setenv("ENTITY_EXTRACTOR_VERSION", "rules-test")

    parser = _build_parser()
    args = parser.parse_args(["--chunk-id", "chunk_1"])

    assert args.extractor_version == "rules-test"


def test_rule_based_extractor_default_wiring_type() -> None:
    """Use deterministic rule-based extractor in entities run flow."""
    extractor = RuleBasedSentenceEntityExtractor()

    assert isinstance(extractor, RuleBasedSentenceEntityExtractor)
