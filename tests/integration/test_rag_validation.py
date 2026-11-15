"""Evaluate hybrid retrieval against the draft parliamentary evidence corpus."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.retrieval.service_layer import retrieve_evidence_chunks
from ddd_policy_tracer.retrieval.sqlite_retrieval import SQLiteHybridRetriever

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "rag_validation_cases.json"
DATABASE_PATH = REPOSITORY_ROOT / "data" / "new3.db"

with VALIDATION_FIXTURE.open(encoding="utf-8") as fixture_file:
    _VALIDATION_CASES: list[dict[str, Any]] = json.load(fixture_file)["cases"]

_ANSWERABLE_CASES = [case for case in _VALIDATION_CASES if case["answerable"]]


@pytest.fixture(scope="module")
def retrieval_components() -> Iterator[tuple[SQLiteHybridRetriever, EmbeddingService]]:
    """Provide real query embeddings and SQLite retrieval for the validation corpus."""
    if os.environ.get("RUN_RAG_VALIDATION") != "1":
        pytest.skip("Set RUN_RAG_VALIDATION=1 to run the model-backed retrieval evaluation.")
    if not DATABASE_PATH.is_file():
        pytest.skip(f"Validation database is not available: {DATABASE_PATH}")

    with SQLiteHybridRetriever(DATABASE_PATH) as retriever:
        yield retriever, EmbeddingService.get_instance()


@pytest.mark.parametrize(
    "case",
    _ANSWERABLE_CASES,
    ids=[case["case_id"] for case in _ANSWERABLE_CASES],
)
def test_retrieval_returns_gold_evidence(
    case: dict[str, Any],
    retrieval_components: tuple[SQLiteHybridRetriever, EmbeddingService],
) -> None:
    """Retrieve every annotated supporting chunk for an answerable validation question."""
    retriever, embedding_service = retrieval_components
    results = retrieve_evidence_chunks(
        query=case["question"],
        retriever=retriever,
        embedding_service=embedding_service,
        limit=10,
    )

    gold_chunk_ids = set(case["gold_evidence_chunk_ids"])
    missing_chunk_ids = gold_chunk_ids - results.keys()
    assert not missing_chunk_ids, (
        f"{case['case_id']} did not retrieve gold evidence {sorted(missing_chunk_ids)}; "
        f"retrieved {list(results)}"
    )
