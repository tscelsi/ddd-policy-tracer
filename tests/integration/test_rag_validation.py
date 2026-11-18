"""Evaluate approved query-to-evidence cases against the indexed speech corpus."""

# ruff: noqa: T201

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.retrieval.evaluation import evaluate_cases
from ddd_policy_tracer.retrieval.sqlite_retrieval import SQLiteHybridRetriever

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "rag_validation_cases.json"
DATABASE_PATH = REPOSITORY_ROOT / "data" / "new4.db"

with VALIDATION_FIXTURE.open(encoding="utf-8") as fixture_file:
    _VALIDATION_CASES: list[dict[str, Any]] = json.load(fixture_file)["cases"]


def test_approved_retrieval_cases_find_all_gold_evidence() -> None:
    """Require every conjunctive target for approved questions in the top ten."""
    if os.environ.get("RUN_RAG_VALIDATION") != "1":
        pytest.skip("Set RUN_RAG_VALIDATION=1 to run the model-backed retrieval evaluation.")
    if not DATABASE_PATH.is_file():
        pytest.skip(f"Validation database is not available: {DATABASE_PATH}")

    with SQLiteHybridRetriever(DATABASE_PATH) as retriever:
        report = evaluate_cases(
            _VALIDATION_CASES,
            retriever=retriever,
            chunk_index=retriever,
            embedding_service=EmbeddingService.get_instance(),
        )

    print(
        f"Approved answerable retrieval cases: {report.answerable_passed}/"
        f"{report.answerable_total} passed.",
    )
    print(
        f"Approved unanswerable cases (reported separately): {len(report.unanswerable_case_ids)}",
    )
    for case in report.cases:
        print(
            f"{case.case_id}: {'PASS' if case.passed else 'FAIL'}; "
            f"target ranks={case.target_ranks}; stale IDs={list(case.stale_chunk_ids)}",
        )
    assert report.answerable_failed == 0
