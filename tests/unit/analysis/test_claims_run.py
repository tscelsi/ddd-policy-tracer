"""Unit tests for claims run module extractor selection behavior."""

from __future__ import annotations

import json
from pathlib import Path

from ddd_policy_tracer.analysis.claims import (
    LLMClaimExtractor,
    OllamaClaimExtractor,
    RuleBasedSentenceClaimExtractor,
)
from ddd_policy_tracer.analysis.claims.run import _build_extractor, _load_chunk_ids


def test_build_extractor_returns_rule_based_strategy() -> None:
    """Build rule extractor when strategy is configured as rule."""
    extractor = _build_extractor(
        extractor_kind="rule",
        rule_threshold=0.9,
        llm_model="gpt-4.1-mini",
        llm_temperature=0.0,
    )

    assert isinstance(extractor, RuleBasedSentenceClaimExtractor)
    assert extractor.config.threshold == 0.9


def test_build_extractor_returns_llm_strategy() -> None:
    """Build LLM extractor when strategy is configured as llm."""
    extractor = _build_extractor(
        extractor_kind="llm",
        rule_threshold=0.8,
        llm_model="gpt-4.1-mini",
        llm_temperature=0.2,
    )

    assert isinstance(extractor, LLMClaimExtractor)
    assert extractor.config.model == "gpt-4.1-mini"
    assert extractor.config.temperature == 0.2


def test_build_extractor_returns_ollama_strategy() -> None:
    """Build Ollama extractor when strategy is configured as ollama."""
    extractor = _build_extractor(
        extractor_kind="ollama",
        rule_threshold=0.8,
        llm_model="llama3.1:8b",
        llm_temperature=0.0,
    )

    assert isinstance(extractor, OllamaClaimExtractor)
    assert extractor.config.model == "llama3.1:8b"


def test_load_chunk_ids_reads_unique_ids_in_file_order(tmp_path: Path) -> None:
    """Return unique chunk ids from chunk JSONL preserving first-seen order."""
    state_path = tmp_path / "chunks.jsonl"
    rows = [
        {"chunk_id": "chunk_1"},
        {"chunk_id": "chunk_2"},
        {"chunk_id": "chunk_1"},
        {"chunk_id": "chunk_3"},
    ]
    with state_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    chunk_ids = _load_chunk_ids(chunk_state_path=state_path)

    assert chunk_ids == ["chunk_1", "chunk_2", "chunk_3"]


def test_load_chunk_ids_returns_empty_list_for_missing_file(tmp_path: Path) -> None:
    """Return empty list when chunk-state file does not exist."""
    state_path = tmp_path / "missing_chunks.jsonl"

    chunk_ids = _load_chunk_ids(chunk_state_path=state_path)

    assert chunk_ids == []
