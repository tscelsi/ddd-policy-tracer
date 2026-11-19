"""Evaluate approved RAG query-to-evidence cases with the configured completion service."""

# ruff: noqa: E402, T201

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from pymongo import AsyncMongoClient

from ddd_policy_tracer.completion.service_layer import RagCompletionService
from ddd_policy_tracer.completion.synthesis.answer_synthesizer import AnswerSynthesizer
from ddd_policy_tracer.app import (
    create_answer_generator,
    create_metadata_filter_extractor,
    load_app_environment,
)
from ddd_policy_tracer.enrichment.embedding import EmbeddingService
from ddd_policy_tracer.ingestion.hansard.talker_repository import MongoTalkerRepository
from ddd_policy_tracer.completion._evaluation.main import (
    EvaluationReport,
    append_evaluation_log,
    evaluate_cases,
)
from ddd_policy_tracer.completion.filtering.resolvers import MongoTalkerNameResolver
from ddd_policy_tracer.completion.retrieval.sqlite_retrieval import SQLiteHybridRetriever

DEFAULT_FIXTURE_PATH = REPOSITORY_ROOT / "scripts" / "rag_retrieval_cases.json"
DEFAULT_DATABASE_PATH = REPOSITORY_ROOT / "data" / "new4.db"
DEFAULT_LOG_PATH = REPOSITORY_ROOT / "logs" / "rag_retrieval_eval.log"


async def _run(
    *,
    database_path: Path,
    fixture_path: Path,
    log_path: Path,
) -> int:
    """Run all approved cases, persist results, and return a failure exit status."""
    with fixture_path.open(encoding="utf-8") as fixture_file:
        cases: list[dict[str, Any]] = json.load(fixture_file)["cases"]

    mongo_client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(
        os.environ.get("MONGODB_URI", "mongodb://localhost:27017"),
    )
    try:
        with SQLiteHybridRetriever(database_path) as retriever:
            completion_service = RagCompletionService(
                retriever=retriever,
                embedding_service=EmbeddingService.get_instance(),
                synthesizer=AnswerSynthesizer(create_answer_generator()),
                metadata_filter_extractor=create_metadata_filter_extractor(),
                speaker_name_resolver=MongoTalkerNameResolver(
                    MongoTalkerRepository(
                        mongo_client,
                        os.environ.get("HANSARD_MONGO_DATABASE", "tmp"),
                    ),
                ),
            )
            report: EvaluationReport = await evaluate_cases(
                cases,
                completion_service=completion_service,
                chunk_index=retriever,
            )
    finally:
        await mongo_client.close()

    append_evaluation_log(report, log_path)
    print(
        f"Approved answerable retrieval cases: {report.answerable_passed}/"
        f"{report.answerable_total} passed.",
    )
    print(
        f"Approved unanswerable cases (reported separately): {len(report.unanswerable_case_ids)}",
    )
    for case in report.cases:
        status = "PASS" if case.passed else "FAIL"
        message = (
            f"{case.case_id}: {status}; target ranks={case.target_ranks}; "
            f"stale IDs={list(case.stale_chunk_ids)}"
        )
        if case.error is not None:
            message += f"; retrieval error={case.error}"
        print(message)
    print(f"Detailed per-case results appended to {log_path}")
    return int(report.answerable_failed > 0)


def main() -> int:
    """Parse evaluation paths, warn about LLM usage, and run the evaluation."""
    load_app_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.environ.get("RAG_DATABASE_PATH", DEFAULT_DATABASE_PATH)),
        help="SQLite chunk database (defaults to RAG_DATABASE_PATH or data/new4.db)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="Reviewed query/evidence fixture file",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(os.environ.get("RAG_VALIDATION_LOG_PATH", DEFAULT_LOG_PATH)),
        help="Append per-case results to this JSON-lines log file",
    )
    args = parser.parse_args()

    if not args.database.is_file():
        parser.error(f"Evaluation database is not available: {args.database}")
    if not args.fixture.is_file():
        parser.error(f"Evaluation fixture is not available: {args.fixture}")

    print(
        "WARNING: each case calls the configured LLM for metadata filtering; "
        "hosted providers may incur usage fees.",
        file=sys.stderr,
    )
    return asyncio.run(
        _run(
            database_path=args.database,
            fixture_path=args.fixture,
            log_path=args.log_file,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
